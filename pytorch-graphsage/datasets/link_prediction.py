from __future__ import division
from math import floor
import os


import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

try:
    from neoConnector import all_cells_with_n_hops_in_area, get_all_edges
except ImportError:
    from .neoConnector import all_cells_with_n_hops_in_area, get_all_edges

class_map = {'inflammatory': 0, 'lymphocyte' : 1, 'fibroblast and endothelial': 2,
               'epithelial': 3, 'apoptosis / civiatte body': 4}

class KIGraphDataset(Dataset):

    def __init__(self, path,  mode='train',
                 num_layers=2,
                 data_split=[0.8, 0.2]):
        """
        Parameters
        ----------
        path : list
            List with filename, coordinates and path to annotation. For example, ['P7_HE_Default_Extended_1_1', (0, 2000, 0, 2000), 'datasets/annotations/P7_annotated/P7_HE_Default_Extended_1_1.txt']
        mode : str
            One of train, val or test. Default: train.
        num_layers : int
            Number of layers in the computation graph. Default: 2.
        data_split: list
            Fraction of edges to use for graph construction / train / val / test. Default: [0.85, 0.08, 0.02, 0.03].
        """
        super().__init__()

        self.path = path
        self.mode = mode
        self.num_layers = num_layers
        self.data_split = data_split

        print('--------------------------------')
        print('Reading dataset from {}'.format(self.path[0]))

        cells, dist = self._read_from_db(self.path) # Get cells and AdjacencyMatrix
        points = self.parse_points(self.path[2]) # Get annotation path points
        coords = np.array([[cell.get('x'), cell.get('y')] for cell in cells]) # Get cell coordinates
        classes = np.array([class_map[cell.get('type')] for cell in cells]) # Get cell classes
        class_scores = np.array([np.array([cell.get('c0'), cell.get('c1'), cell.get('c2'), cell.get('c3')]) for cell in cells]) # Get cell classes
        dist = np.array(dist)
        adj = np.copy(dist)
        adj[adj != 0] = 1
        adj_edge = np.array(adj_to_edge(adj.tolist())) # Get neighbors on edge list format

        _, edges_all = get_all_edges(self.path[0]+"_final", self.path[1], hops=2)
        edges_all = np.array(adj_to_edge(edges_all))

        self.dist = dist
        self.classes = classes
        self.class_scores = class_scores
        self.coords = coords

        print('Finished reading data.')

        print('Setting up graph.')
        vertex_id = {j : i for (i, j) in enumerate(range(len(coords)))}

        idxs = [floor(v*edges_all.shape[0]) for v in np.cumsum(data_split)]

        edges_t, pos_examples = adj_edge, edges_all # edges_t = all neighbor edges, pos_examples = all edges passing boundary

        edges_t[:, :2] = np.array([vertex_id[u] for u in edges_t[:, :2].flatten()]).reshape(edges_t[:, :2].shape)
        edges_s = np.unique(edges_t[:, :2], axis=0) # Filter duplicate edges

        self.n = len(vertex_id) # Count vertices
        self.m_s= edges_s.shape[0] # Count edges

        adj = sp.coo_matrix((np.ones(self.m_s), (edges_s[:, 0], edges_s[:, 1])),
                            shape=(self.n,self.n),
                            dtype=np.float32)
        # Symmetric.
        #adj += adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        self.adj = adj.tolil()

        self.nbrs_s = self.adj.rows # Neighbors

        #print(self.nbrs_s[0])
        #print(self.dist[0,:])
        #exit()
        # One hot encode all class labels
        features = np.zeros((classes.size, classes.max()+1))
        features[np.arange(classes.size),classes] = 1
        features = np.concatenate((features, class_scores), axis=1)
        self.features = torch.from_numpy(features).float() # Cell features

        print('Finished setting up graph.')

        print('Setting up examples.')

        if len(pos_examples) > 0:
            pos_examples = pos_examples[:, :2]
            pos_examples = np.unique(pos_examples, axis=0)

        # Generate negative examples not in cell edges crossing path
        neg_examples = []
        cur = 0
        n, _choice = self.n, np.random.choice
        neg_seen = set(tuple(e[:2]) for e in edges_all) # Dont sample positive edges
        adj_tuple = set(tuple(e[:2]) for e in adj_edge) # List all edges

        if self.mode != 'train':    # Add all edges except positive edges if validation/test

            for example in adj_edge:
                if (example[0], example[1]) in neg_seen:
                    continue
                neg_examples.append(example)
            neg_examples = np.array(neg_examples, dtype=np.int64)
            #if self.mode == 'train':
                #while(pos_examples.shape[0] < neg_examples.shape[0]):   # Oversample positive examples
                #    pos_examples = np.append(pos_examples, pos_examples[:pos_len, :], axis=0)
        else:   # Undersample negative samples from adjacency edges not in positive
            num_neg_examples = pos_examples.shape[0]
            while cur < num_neg_examples:
                u, v = _choice(n, 2, replace=False)
                if (u, v) in neg_seen or (u, v) not in adj_tuple:
                    continue
                cur += 1
                neg_examples.append([u, v])
            neg_examples = np.array(neg_examples, dtype=np.int64)



        x = np.vstack((pos_examples, neg_examples))
        y = np.concatenate((np.ones(pos_examples.shape[0]),
                            np.zeros(neg_examples.shape[0])))
        perm = np.random.permutation(x.shape[0])
        x, y = x[perm, :], y[perm]
        x, y = torch.from_numpy(x).long(), torch.from_numpy(y).long()
        self.x, self.y = x, y

        print('Finished setting up examples.')

        print('Dataset properties:')
        print('Mode: {}'.format(self.mode))
        print('Number of vertices: {}'.format(self.n))
        print('Number of edges: {}'.format(self.m_s))
        print('Number of positive/negative datapoints: {}/{}'.format(pos_examples.shape[0],neg_examples.shape[0]))
        print('Number of examples/datapoints: {}'.format(self.x.shape[0]))
        print('--------------------------------')


    def _read_from_db(self, path):
        cells, dist = all_cells_with_n_hops_in_area(path[0]+"_final", path[1], hops=2)
        return cells, dist

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def get_coords_and_class(self):
        return self.coords, self.classes

    def _form_computation_graph(self, idx):
        """
        Parameters
        ----------
        idx : int or list
            Indices of the node for which the forward pass needs to be computed.
        Returns
        -------
        node_layers : list of numpy array
            node_layers[i] is an array of the nodes in the ith layer of the
            computation graph.
        mappings : list of dictionary
            mappings[i] is a dictionary mapping node v (labelled 0 to |V|-1)
            in node_layers[i] to its position in node_layers[i]. For example,
            if node_layers[i] = [2,5], then mappings[i][2] = 0 and
            mappings[i][5] = 1.
        """
        _list, _set = list, set
        if type(idx) is int:
            node_layers = [np.array([idx], dtype=np.int64)]
        elif type(idx) is list:
            node_layers = [np.array(idx, dtype=np.int64)]

        for _ in range(self.num_layers):
            prev = node_layers[-1]
            arr = [node for node in prev]
            arr.extend([e for node in arr for e in self.nbrs_s[node]])  # add neighbors to graph
            arr = np.array(_list(_set(arr)), dtype=np.int64)
            node_layers.append(arr)
        node_layers.reverse()

        mappings = [{j : i for (i, j) in enumerate(arr)} for arr in node_layers]

        return node_layers, mappings

    def collate_wrapper(self, batch):
        """
        Parameters
        ----------
        batch : list
            A list of examples from this dataset. An example is (edge, label).
        Returns
        -------
        edges : numpy array
            The edges in the batch.
        features : torch.FloatTensor
            An (n' x input_dim) tensor of input node features.
        node_layers : list of numpy array
            node_layers[i] is an array of the nodes in the ith layer of the
            computation graph.
        mappings : list of dictionary
            mappings[i] is a dictionary mapping node v (labelled 0 to |V|-1)
            in node_layers[i] to its position in node_layers[i]. For example,
            if node_layers[i] = [2,5], then mappings[i][2] = 0 and
            mappings[i][5] = 1.
        rows : numpy array
            Each row is the list of neighbors of nodes in node_layers[0].
        labels : torch.LongTensor
            Labels (1 or 0) for the edges in the batch.
        """
        idx = list(set([v.item() for sample in batch for v in sample[0][:2]]))

        node_layers, mappings = self._form_computation_graph(idx)

        rows = self.nbrs_s[node_layers[0]]
        features = self.features[node_layers[0], :]

        dist = torch.from_numpy(self.dist)
        labels = torch.FloatTensor([sample[1] for sample in batch])
        edges = np.array([sample[0].numpy() for sample in batch])
        edges = np.array([mappings[-1][v] for v in edges.flatten()]).reshape(edges.shape)

        # TODO: Pin memory. Change type of node_layers, mappings and rows to
        # tensor?

        return edges, features, node_layers, mappings, rows, labels, dist

    def get_dims(self):
        return self.features.shape[1], 1

    def parse_points(self, fname):
        with open(fname, 'r') as f:
            lines = f.readlines()
        lines = [line[:-1].split(',') for line in lines] # Remove \n from line
        return lines

    class KIGraphDataset2(Dataset):

        def __init__(self, path, mode='train',
                     num_layers=2,
                     data_split=[0.8, 0.2], add_self_edges=False):
            """
            Parameters
            ----------
            path : list
                List with filename, coordinates and path to annotation. For example, ['P7_HE_Default_Extended_1_1', (0, 2000, 0, 2000), 'datasets/annotations/P7_annotated/P7_HE_Default_Extended_1_1.txt']
            mode : str
                One of train, val or test. Default: train.
            num_layers : int
                Number of layers in the computation graph. Default: 2.
            data_split: list
                Fraction of edges to use for graph construction / train / val / test. Default: [0.85, 0.08, 0.02, 0.03].
            """
            super().__init__()

            self.path = path
            self.mode = mode
            self.num_layers = num_layers
            self.data_split = data_split

            print('--------------------------------')
            print('Reading dataset from {}'.format(self.path[0]))

            # cells, distances_close_to_edges = self._read_from_db(self.path)  # Get cells and AdjacencyMatrix
            # points = self.parse_points(self.path[2])  # Get annotation path points
            # coords = np.array([[cell.get('x'), cell.get('y')] for cell in cells])  # Get cell coordinates
            # all_labels_cell_types = np.array([class_map[cell.get('type')] for cell in cells])  # Get cell classes
            # cell_types_scores = np.array([np.array([cell.get('c0'), cell.get('c1'), cell.get('c2'), cell.get('c3')]) for cell in cells]) # Get cell classes
            # distances_close_to_edges = np.array(distances_close_to_edges)
            # adjacency_matrix_close_to_edges = np.copy(distances_close_to_edges)
            # adjacency_matrix_close_to_edges[adjacency_matrix_close_to_edges != 0] = 1
            # edge_list_close_to_edge = np.array(adj_to_edge(adjacency_matrix_close_to_edges.tolist()))  # Get neighbors on edge list format
            #
            # _, adjacency_matrix_crossing_edges = get_all_edges(self.path[0]+"_final", self.path[1], hops=2)
            # edge_list_crossing_edges = np.array(adj_to_edge(adjacency_matrix_crossing_edges))

            ########## MINE ###########
            # Cells, distance_close_to_edges
            # edge_path = "2021-4-5_19_44_P9_1_1_edges_intelligraph_idx.csv"
            edge_path = path[1]
            # node_path = "2021-4-5_19_44_P9_1_1_nodes_intelligraph_idx.csv"
            node_path = path[2]

            edges = pd.read_csv(f'.\ki\{edge_path}')
            nodes = pd.read_csv(f'.\ki\{node_path}')

            if add_self_edges:
                for i in range(len(nodes)):
                    new_row = {'source': i, 'target': i, 'type': 0, 'distance': 0}
                    # append row to the dataframe
                    edges = edges.append(new_row, ignore_index=True)

            edges_crossing = edges.copy()
            edges_crossing = edges_crossing[edges_crossing["type"] == 1]

            edges['type'] = edges['type'].replace(1, 0)

            col_row_len = len(nodes['id'])
            distances_close_to_edges = pd.DataFrame(0, index=np.arange(col_row_len), columns=np.arange(col_row_len))

            for i, row in edges.iterrows():
                source = row['source']
                target = row['target']
                distance = row['distance']
                distances_close_to_edges[source][target] = distance
                distances_close_to_edges[target][source] = distance

            distances_close_to_edges = np.array(distances_close_to_edges)

            # coords
            coords = nodes[["x", "y"]].to_numpy()

            # all_labels_cell_types
            nodes["type"].replace({'inflammatory': 0, 'lymphocyte': 1, 'fibroblast and endothelial': 2,
                                   'epithelial': 3}, inplace=True)

            all_labels_cell_types = nodes["type"].to_numpy()
            nodes_with_types_zero_one = nodes.copy()
            for i, row in nodes.iterrows():
                # if row['gt'] == 'null':
                nodes_with_types_zero_one.iloc[i, 3] = 1 if row['type'] == 2 else 0
                nodes_with_types_zero_one.iloc[i, 4] = 1 if row['type'] == 0 else 0
                nodes_with_types_zero_one.iloc[i, 1] = 1 if row['type'] == 1 else 0
                nodes_with_types_zero_one.iloc[i, 2] = 1 if row['type'] == 3 else 0
            # else:
            #     nodes_with_types_zero_one.iloc[i, 3] = 1 if row['gt'] == 'fibroblast and endothelial' else 0
            #     nodes_with_types_zero_one.iloc[i, 4] = 1 if row['gt'] == 'inflammatory' else 0
            #     nodes_with_types_zero_one.iloc[i, 1] = 1 if row['gt'] == 'lymphocyte' else 0
            #     nodes_with_types_zero_one.iloc[i, 2] = 1 if row['gt'] == 'epithelial' else 0

            # cell_types_scores
            cell_types_scores = nodes_with_types_zero_one[['inf', 'lym', 'fib', 'epi']]
            cell_types_scores = cell_types_scores.to_numpy()

            # adjacency_matrix_close_to_edges
            adjacency_matrix_close_to_edges = np.copy(distances_close_to_edges)
            adjacency_matrix_close_to_edges[adjacency_matrix_close_to_edges != 0] = 1

            # edge_list_close_to_edge
            edge_list_close_to_edge = edges[["source", "target"]]
            edge_list_close_to_edge = edge_list_close_to_edge.to_numpy()

            # edge_list_crossing_edges TODO
            edge_list_crossing_edges = edges_crossing.to_numpy()

            self.am_close_to_edges_including_distances = distances_close_to_edges
            self.classes = all_labels_cell_types
            self.class_scores = cell_types_scores
            self.coords = coords

            print('Finished reading data.')

            print('Setting up graph.')
            vertex_id = {j: i for (i, j) in enumerate(range(len(coords)))}

            idxs = [floor(v * edge_list_crossing_edges.shape[0]) for v in np.cumsum(data_split)]

            edges_t, pos_examples_crossing_edges = edge_list_close_to_edge, edge_list_crossing_edges
            # print("Edges_t.shape = {}".format(edges_t.shape))
            # edges_t = all neighbor edges, pos_examples = all edges passing boundary

            edges_t[:, :2] = np.array([vertex_id[u] for u in edges_t[:, :2].flatten()]).reshape(edges_t[:, :2].shape)
            edges_t_no_duplicates = np.unique(edges_t[:, :2], axis=0)  # Filter duplicate edges

            self.nodes_count = len(vertex_id)  # Count vertices
            self.edges_count = edges_t_no_duplicates.shape[0]  # Count edges

            adjacency_matrix_close_to_edges = sp.coo_matrix(
                (np.ones(self.edges_count), (edges_t_no_duplicates[:, 0], edges_t_no_duplicates[:, 1])),
                shape=(self.nodes_count, self.nodes_count),
                dtype=np.float32)
            # Symmetric.
            # adj += adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

            self.adjacency_matrix_close_to_edges_as_coo_to_lil = adjacency_matrix_close_to_edges.tolil()

            self.node_neighbors = self.adjacency_matrix_close_to_edges_as_coo_to_lil.rows  # Neighbors

            # print(self.nbrs_s[0])
            # print(self.dist[0,:])
            # exit()
            # One hot encode all class labels
            one_hot_encoded_features_cell_types = np.zeros(
                (all_labels_cell_types.size, all_labels_cell_types.max() + 1))
            one_hot_encoded_features_cell_types[np.arange(all_labels_cell_types.size), all_labels_cell_types] = 1
            one_hot_encoded_features_cell_types = np.concatenate(
                (one_hot_encoded_features_cell_types, cell_types_scores), axis=1)
            self.features = torch.from_numpy(one_hot_encoded_features_cell_types).float()  # Cell features

            print('Finished setting up graph.')

            print('Setting up examples.')

            if len(pos_examples_crossing_edges) > 0:
                pos_examples_crossing_edges = pos_examples_crossing_edges[:, :2]
                pos_examples_crossing_edges = np.unique(pos_examples_crossing_edges, axis=0)

            # Generate negative examples not in cell edges crossing path
            neg_examples_close_to_edges = []
            cur = 0
            n_count, _choice = self.nodes_count, np.random.choice
            neg_seen = set(tuple(e[:2]) for e in edge_list_crossing_edges)  # Dont sample positive edges
            adj_tuple = set(tuple(e[:2]) for e in edge_list_close_to_edge)  # List all edges

            if self.mode != 'train':  # Add all edges except positive edges if validation/test
                print("self.mode != 'train'")
                for example in edge_list_close_to_edge:
                    if (example[0], example[1]) in neg_seen:
                        continue
                    neg_examples_close_to_edges.append(example)
                neg_examples_close_to_edges = np.array(neg_examples_close_to_edges, dtype=np.int64)
                # if self.mode == 'train':
                # while(pos_examples.shape[0] < neg_examples.shape[0]):   # Oversample positive examples
                #    pos_examples = np.append(pos_examples, pos_examples[:pos_len, :], axis=0)
            else:  # Undersample negative samples from adjacency edges not in positive
                num_neg_examples = pos_examples_crossing_edges.shape[0]
                while cur < num_neg_examples:
                    u, v = _choice(n_count, 2, replace=False)
                    if (u, v) in neg_seen or (u, v) not in adj_tuple:
                        continue
                    cur += 1
                    neg_examples_close_to_edges.append([u, v])
                neg_examples_close_to_edges = np.array(neg_examples_close_to_edges, dtype=np.int64)

            x = np.vstack((pos_examples_crossing_edges, neg_examples_close_to_edges))
            y = np.concatenate((np.ones(pos_examples_crossing_edges.shape[0]),
                                np.zeros(neg_examples_close_to_edges.shape[0])))
            perm = np.random.permutation(x.shape[0])
            # print("x.shape[0] = {}, x.shape = {},y.shape = {}".format(x.shape[0], x.shape, y.shape))
            # print("perm = {}".format(perm))
            x, y = x[perm, :], y[perm]  # ERROR HERE -> IndexError: too many indices for array: array is 1-dimensional,
            # but 2 were indexed
            x, y = torch.from_numpy(x).long(), torch.from_numpy(y).long()
            self.x, self.y = x, y

            print('Finished setting up examples.')

            print('Dataset properties:')
            print('Mode: {}'.format(self.mode))
            print('Number of vertices: {}'.format(self.nodes_count))
            print('Number of edges: {}'.format(self.edges_count))
            print('Number of positive/negative datapoints: {}/{}'.format(pos_examples_crossing_edges.shape[0],
                                                                         neg_examples_close_to_edges.shape[0]))
            print('Number of examples/datapoints: {}'.format(self.x.shape[0]))
            print('--------------------------------')

        def _read_from_db(self, path):
            cells, dist = all_cells_with_n_hops_in_area(path[0] + "_final", path[1], hops=2)
            return cells, dist

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

        def get_coords_and_class(self):
            return self.coords, self.classes

        def _form_computation_graph(self, idx):
            """
            Parameters
            ----------
            idx : int or list
                Indices of the node for which the forward pass needs to be computed.
            Returns
            -------
            node_layers : list of numpy array
                node_layers[i] is an array of the nodes in the ith layer of the
                computation graph.
            mappings : list of dictionary
                mappings[i] is a dictionary mapping node v (labelled 0 to |V|-1)
                in node_layers[i] to its position in node_layers[i]. For example,
                if node_layers[i] = [2,5], then mappings[i][2] = 0 and
                mappings[i][5] = 1.
            """
            _list, _set = list, set
            if type(idx) is int:
                node_layers = [np.array([idx], dtype=np.int64)]
            elif type(idx) is list:
                node_layers = [np.array(idx, dtype=np.int64)]

            for _ in range(self.num_layers):
                prev = node_layers[-1]
                arr = [node for node in prev]
                arr.extend([e for node in arr for e in self.node_neighbors[node]])  # add neighbors to graph
                arr = np.array(_list(_set(arr)), dtype=np.int64)
                node_layers.append(arr)
            node_layers.reverse()

            mappings = [{j: i for (i, j) in enumerate(arr)} for arr in node_layers]

            return node_layers, mappings

        def collate_wrapper(self, batch):
            """
            Parameters
            ----------
            batch : list
                A list of examples from this dataset. An example is (edge, label).
            Returns
            -------
            edges : numpy array
                The edges in the batch.
            features : torch.FloatTensor
                An (n' x input_dim) tensor of input node features.
            node_layers : list of numpy array
                node_layers[i] is an array of the nodes in the ith layer of the
                computation graph.
            mappings : list of dictionary
                mappings[i] is a dictionary mapping node v (labelled 0 to |V|-1)
                in node_layers[i] to its position in node_layers[i]. For example,
                if node_layers[i] = [2,5], then mappings[i][2] = 0 and
                mappings[i][5] = 1.
            rows : numpy array
                Each row is the list of neighbors of nodes in node_layers[0].
            labels : torch.LongTensor
                Labels (1 or 0) for the edges in the batch.
            """
            idx = list(set([v.item() for sample in batch for v in sample[0][:2]]))

            node_layers, mappings = self._form_computation_graph(idx)

            rows = self.node_neighbors[node_layers[0]]
            features = self.features[node_layers[0], :]

            dist = torch.from_numpy(self.am_close_to_edges_including_distances)
            labels = torch.FloatTensor([sample[1] for sample in batch])
            edges = np.array([sample[0].numpy() for sample in batch])
            edges = np.array([mappings[-1][v] for v in edges.flatten()]).reshape(edges.shape)

            # TODO: Pin memory. Change type of node_layers, mappings and rows to
            # tensor?

            return edges, features, node_layers, mappings, rows, labels, dist

        def get_dims(self):
            print("self.features.shape: {}".format(self.features.shape))
            print("input_dims (input dimension) -> self.features.shape[1] = {}".format(self.features.shape[1]))
            return self.features.shape[1], 1

        def parse_points(self, fname):
            with open(fname, 'r') as f:
                lines = f.readlines()
            lines = [line[:-1].split(',') for line in lines]  # Remove \n from line
            return lines

def adj_to_edge(adj):
    edges = []
    for i in range(len(adj)):
        edges += ([[i,index] for index, element in enumerate(adj[i]) if element == 1])

    return edges

def get_intersections(points, coords, adj):
    # Loop through cells
    intersections = []
    count = 0
    for i in range(len(coords)):
        # Get ids of all neighbors
        nbrs = [index for index, element in enumerate(adj[i]) if element == 1]
        for j in range(len(nbrs)):
            passed = False
            for k in range(len(points)-2):
                if len(points[k]) == 2 and len(points[k+1]) == 2:
                    L1 = line(coords[i], coords[nbrs[j]]) # Line between node and neighbor
                    L2 = line([int(float(point)) for point in points[k]], [int(float(point)) for point in points[k+1]]) # Line between two points of path
                    inter = intersection(L1, L2) # Get x-coordinate for intersection or False if none
                    if inter != False:
                        if ( (inter > max( min(coords[i][0],coords[nbrs[j]][0]), min(int(float(points[k][0])),int(float(points[k+1][0]))) )) and
                            (inter < min( max(coords[i][0],coords[nbrs[j]][0]), max(int(float(points[k][0])),int(float(points[k+1][0]))) )) ): # If intersection is inside line segments
                            intersections.append([i, nbrs[j]])
                            passed = True
                            break
            #if passed == False: # If no intersections between cell and neighbor
            #    intersections.append([i, nbrs[j], 0])
            #
    return intersections

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    if D != 0:
        x = Dx / D
        return x
    else:
        return False
