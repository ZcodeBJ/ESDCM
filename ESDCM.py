from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity, normalize
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import MessagePassing
from torch.autograd import Variable
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from loss import *








class personaGATConvLayer(MessagePassing):
    def __init__(self, in_channels, persona_channels, out_channels, 
                 heads=1, 
                 concat=True, 
                 bias=True,
                 negative_slope= 0.2,
                 dropout=0.6,
                 training=True):
        super(personaGATConvLayer, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.bias = bias
        self.negative_slope = negative_slope
        self.training = training
        self.persona_gate = nn.ModuleList([Linear(in_channels, 1, weight_initializer='glorot') for _ in range(heads)])
        self.lins_persona = nn.ModuleList([Linear(persona_channels, int(out_channels/heads), bias=False, weight_initializer='glorot') for _ in range(heads)])
        self.lins = nn.ModuleList([Linear(in_channels, int(out_channels/heads), bias=False, weight_initializer='glorot') for _ in range(heads)])
        self.attentions = nn.ModuleList([Linear(int(out_channels/heads)*2, 1, bias=False, weight_initializer='glorot') for _ in range(heads)])
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()    

    def reset_parameters(self):
        super().reset_parameters()
        for head in range(self.heads):
            self.persona_gate[head].reset_parameters()
            self.lins_persona[head].reset_parameters()
            self.lins[head].reset_parameters()
            self.attentions[head].reset_parameters()
            zeros(self.bias)


    def forward(self, x, persona,edge_index, size=None):
        # x = F.dropout(x, p=0.6, training=self.training)
        persona_features = []
        x_features = []

        for head in range(self.heads):
            persona_features.append(self.persona_gate[head](x) * self.lins_persona[head](persona))
            x_features.append(self.lins[head](x))

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))


        scores = []
        for head in range(self.heads):
            persona_i = persona_features[head][edge_index[0]]
            persona_j = persona_features[head][edge_index[1]]

            score = self.attentions[head](torch.cat([persona_i, persona_j], dim=-1))
            score = F.leaky_relu(score, negative_slope=self.negative_slope)
            score = softmax(score, edge_index[1], num_nodes=size)
            scores.append(score)


        out_features = []
        for head in range(self.heads):
            out = self.propagate(edge_index, x=x_features[head], norm=scores[head])
            out_features.append(out)

        out = torch.cat(out_features, dim=-1)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out






class personaGAT(nn.Module):
    def __init__(self, in_channels, persona_channels, out_channels, heads, layer_nums, dropout, training=True):
        super(personaGAT, self).__init__()
        self.layer_nums = layer_nums
        self.dropout = dropout
        self.training = training
        self.GATConv = nn.ModuleList([personaGATConvLayer(in_channels=in_channels, persona_channels=persona_channels, out_channels=out_channels, heads=heads)] \
                                     + [personaGATConvLayer(in_channels=out_channels, persona_channels=persona_channels, out_channels=out_channels, heads=heads) for _ in range(self.layer_nums-1)])

    def forward(self, x, persona, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.layer_nums-1):
            out = F.dropout(self.GATConv[i](x, persona, edge_index), p=self.dropout, training=self.training)
            out = F.elu(out)
            x = x + out
            # x = F.elu(self.GATConv[i](x, persona, edge_index))
            # x = F.dropout(x, p=self.dropout, training=self.training)
        out = F.dropout(self.GATConv[-1](x, persona, edge_index), p=self.dropout, training=self.training)
        out = F.elu(out)
        x = x + out
        return x












class interactiveGAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads, layer_nums, dropout=0.6, training=True):
        super(interactiveGAT, self).__init__()
        self.layer_nums = layer_nums
        self.training = training
        self.dropout = dropout
        self.GATConv = nn.ModuleList([RealtionalGATConv(in_channels, out_channels, heads) for _ in range(self.layer_nums-1)] + \
                                     [RealtionalGATConv(out_channels, out_channels, heads)])

    def forward(self, x, edge_index, edge_indice, edge_type, edge_dialog):

        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.layer_nums-1):
            out = F.dropout(self.GATConv[i](x, edge_index, edge_indice, edge_type, edge_dialog), p=self.dropout, training=self.training)
            out = F.elu(out)
            x = x + out

            # x = F.elu(self.GATConv[i](x, edge_index, edge_type, edge_dialog))
            # x = F.dropout(x, p=self.dropout, training=self.training)
        out = F.dropout(self.GATConv[-1](x, edge_index, edge_indice, edge_type, edge_dialog), p=self.dropout, training=self.training)
        out = F.elu(out)
        x = x + out

        # x = self.GATConv[-1](x, edge_index, edge_type, edge_dialog)
        # x.retain_grad()  
        return x



class RealtionalGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, relation_nums=8,
                 concat=True, bias=True, negative_slope= 0.2, dropout=0.6, training=True):
        super(RealtionalGATConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.bias = bias
        self.negative_slope = negative_slope
        self.training = training
        self.position_encodings_weight = Parameter(torch.Tensor(relation_nums, heads)) 
        self.position_encodings_bias = Parameter(torch.Tensor(relation_nums, heads)) 
        self.lins = nn.ModuleList([Linear(in_channels, int(out_channels/heads), bias=False, weight_initializer='glorot') for _ in range(heads)])
        self.attentions = nn.ModuleList([Linear(int(out_channels/heads)*2, 1, bias=False, weight_initializer='glorot') for _ in range(heads)])
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()    
        


    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.position_encodings_weight)
        glorot(self.position_encodings_bias)
        for head in range(self.heads):
            self.lins[head].reset_parameters()
            self.attentions[head].reset_parameters()
            zeros(self.bias)

    def normalize_tensor_by_labels(self, tensor, labels):
        unique_labels = labels.unique()
        normalized_tensor = torch.zeros_like(tensor)

        for label in unique_labels:
            indices = (labels == label).nonzero(as_tuple=True)[0]
            label_data = tensor[indices]
            normalized_label_data = (label_data - label_data.mean()) / label_data.std()
         
            normalized_tensor[indices] = normalized_label_data

        return normalized_tensor



    def normalize_tensor_by_labels(self, tensor, edge_type, edge_dialog):
        unique_types = edge_type.unique()
        normalized_tensor = torch.zeros_like(tensor).float()

        for dia_index in range(int(torch.max(edge_dialog))+1):
            dia_uttr_indices = torch.where(edge_dialog == dia_index)[0]

            for type in unique_types:
                type_indices = (edge_type == type).nonzero(as_tuple=True)[0]


                mask = torch.isin(dia_uttr_indices, type_indices)
                intersection_indices = dia_uttr_indices[mask]



        return normalized_tensor





    def forward(self, x, edge_index, edge_indice, edge_type, edge_dialog, size=None):

        x = F.dropout(x, p=self.dropout, training=self.training)
        out_features = []

        for head in range(self.heads):

            x_features = self.lins[head](x)

            score = self.attentions[head](torch.cat([x_features[edge_index[0]], x_features[edge_index[1]]], dim=-1))
            score = F.leaky_relu(score, negative_slope=self.negative_slope)

            #
            # score = torch.where(score > self.threshold, score, torch.tensor(0.0).to(score.device))

            score = softmax(score, edge_index[1], num_nodes=size)
            out = self.propagate(edge_index, x=x_features, norm=score)
            out_features.append(out)

        out = torch.cat(out_features, dim=-1)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out




class GATModel(nn.Module):

    def __init__(self, base_model: object, D_m: object, D_m_v: object, D_m_a: object, n_speakers: object,
                 graph_type: object = 'relation',
                 modals: object = 'avl', dataset: object = 'IEMOCAP',
                 speaker_weights: object = '1-1-1',
                 hidden_l: object = 300, hidden_a: object = 300, hidden_v: object = 300,
                 persona_l_heads: object = 4, persona_a_heads: object = 4, persona_v_heads: object = 4,
                 persona_l_layer: object = 1, persona_a_layer: object = 1, persona_v_layer: object = 1,
                 interactive_layer: object = 2, interactive_heads: object = 4, persona_transform: object = False,
                 dropout_forward: object = 0.5, dropout_persona_lstm_modeling: object = 0.5, dropout_interactive: object = 0.1,
                 dropout_persona: object = 0.1,
                 dropout_smax_erc: object = 0.5, dropout_smax_shift: object = 0.5,
                 erc_windows: object = 1, shift_windows: object = 1, interactive_windows: object = 1,
                 av_using_lstm: object = False,
                 norm: object = 'BN',
                 wo_persona: object = False, wo_crosstask: object = False, wo_shiftcl: object = False, wo_emotioncl: object = False, wo_ercByShiftEdges: object = False,
                 wo_pareto: object = False,
                 weight: object = 2,
                 ) -> object:

        super(GATModel, self).__init__()

        self.modals = [x for x in modals]  # [a, v, l]
        self.n_speakers = n_speakers
        self.speaker_weights = list(map(float, speaker_weights.split('-')))

        
        self.class_num = 6 if dataset == 'IEMOCAP' else 7
        
        self.use_bert_seq = False
        self.dataset = dataset
        self.wo_persona = wo_persona
        self.wo_crosstask = wo_crosstask
        self.wo_shiftcl = wo_shiftcl
        self.wo_emotioncl = wo_emotioncl
        self.wo_ercByShiftEdges = wo_ercByShiftEdges
        self.wo_pareto= wo_pareto
        self.weight = weight

        hidden_l = hidden_l
        hidden_a = hidden_a
        hidden_v = hidden_v

        self.dropout_forward = nn.Dropout(dropout_forward)

        #################
        # norm stragy
        self.norm_strategy = norm
        if self.norm_strategy == 'BN':
            self.normBNa = nn.BatchNorm1d(D_m, affine=True)
            self.normBNb = nn.BatchNorm1d(D_m, affine=True)
            self.normBNc = nn.BatchNorm1d(D_m, affine=True)
            self.normBNd = nn.BatchNorm1d(D_m, affine=True)
        elif self.norm_strategy == 'LN':
            self.normLNa = nn.LayerNorm(D_m, elementwise_affine=True)
            self.normLNb = nn.LayerNorm(D_m, elementwise_affine=True)
            self.normLNc = nn.LayerNorm(D_m, elementwise_affine=True)
            self.normLNd = nn.LayerNorm(D_m, elementwise_affine=True)

        ##################
        # persona modeling
        self.linear_l = nn.Linear(D_m, hidden_l)
        self.linear_a = nn.Linear(D_m_a, hidden_a)
        self.linear_v = nn.Linear(D_m_v, hidden_v)

        self.lstm_l = nn.GRU(input_size=hidden_l, hidden_size=int(hidden_l/2), num_layers=2, bidirectional=True, dropout=dropout_persona_lstm_modeling)
        self.speaker_weights_l1 = nn.Linear(hidden_l, hidden_l)
        self.speaker_weights_l2 = nn.Linear(hidden_l, hidden_l)
        self.speaker_weights_a1 = nn.Linear(hidden_a, hidden_a)
        self.speaker_weights_a2 = nn.Linear(hidden_a, hidden_a)
        self.speaker_weights_v1 = nn.Linear(hidden_v, hidden_v)
        self.speaker_weights_v2 = nn.Linear(hidden_v, hidden_v)
        self.av_using_lstm = av_using_lstm
        if self.av_using_lstm:
            self.lstm_a = nn.GRU(input_size=hidden_a, hidden_size=int(hidden_a/2), num_layers=2, bidirectional=True, dropout=dropout_persona_lstm_modeling)
            self.lstm_v = nn.GRU(input_size=hidden_v, hidden_size=int(hidden_v/2), num_layers=2, bidirectional=True, dropout=dropout_persona_lstm_modeling)

        self.rnn_parties_l = nn.GRU(input_size=hidden_l, hidden_size=int(hidden_l/2), num_layers=2, bidirectional=True, dropout=dropout_persona_lstm_modeling)
        self.rnn_parties_a = nn.GRU(input_size=hidden_a, hidden_size=int(hidden_a/2), num_layers=2, bidirectional=True, dropout=dropout_persona_lstm_modeling)
        self.rnn_parties_v = nn.GRU(input_size=hidden_v, hidden_size=int(hidden_v/2), num_layers=2, bidirectional=True, dropout=dropout_persona_lstm_modeling)


        ##################
        # persona inject
        if self.wo_persona == False:
            self.persona_hidden = 5
            self.persona_transform = persona_transform
            if persona_transform:
                self.persona_hidden = hidden_l
                self.linear_persona = nn.Linear(5, self.persona_hidden)
            
            if 'l' in self.modals:
                self.personaGAT_l = personaGAT(in_channels=hidden_l, persona_channels=self.persona_hidden, out_channels=hidden_l, heads=persona_l_heads, layer_nums=persona_l_layer, dropout=dropout_persona)
            if 'a' in self.modals:
                self.personaGAT_a = personaGAT(in_channels=hidden_a, persona_channels=self.persona_hidden, out_channels=hidden_a, heads=persona_a_heads, layer_nums=persona_a_layer, dropout=dropout_persona)
            if 'v' in self.modals:
                self.personaGAT_v = personaGAT(in_channels=hidden_v, persona_channels=self.persona_hidden, out_channels=hidden_v, heads=persona_v_heads, layer_nums=persona_v_layer, dropout=dropout_persona)
            self.persona_transformation_l1 = nn.Linear(hidden_l, hidden_l)
            self.persona_transformation_l2 = nn.Linear(hidden_l, hidden_l)
            self.persona_transformation_a1 = nn.Linear(hidden_a, hidden_a)
            self.persona_transformation_a2 = nn.Linear(hidden_a, hidden_a)
            self.persona_transformation_v1 = nn.Linear(hidden_v, hidden_v)
            self.persona_transformation_v2 = nn.Linear(hidden_v, hidden_v)
        

        ###################
        # cross task interactive
        self.erc_windows = erc_windows                   ### min erc_windows = 1
        self.shift_windows = shift_windows               ### min shift_windows = 1
        self.interactive_windows = interactive_windows   ### min interactive_windows = 0
        self.interactive_ES_GAT = interactiveGAT(in_channels=hidden_l, out_channels=hidden_l, heads=interactive_heads, layer_nums=interactive_layer, dropout=dropout_interactive)
        self.interactive_ERC_GAT = interactiveGAT(in_channels=hidden_l, out_channels=hidden_l, heads=interactive_heads, layer_nums=interactive_layer, dropout=dropout_interactive)

        self.interactive_ES_transformation1 = nn.Linear(hidden_l, hidden_l)
        self.interactive_ES_transformation2 = nn.Linear(hidden_l, hidden_l)

        self.interactive_ERC_transformation1 = nn.Linear(hidden_l, hidden_l)
        self.interactive_ERC_transformation2 = nn.Linear(hidden_l, hidden_l)

        ###################
        # smax
        self.smax_fc_ERC = nn.Linear(len(modals)*hidden_l, self.class_num)
        self.dropout_smax_erc = nn.Dropout(p=dropout_smax_erc)
        self.smax_fc_ES = nn.Linear(2*len(modals)*hidden_l, self.class_num**2)
        self.dropout_smax_shift = nn.Dropout(p=dropout_smax_shift)
        self.reset_parameters()



    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_l.weight)
        nn.init.xavier_uniform_(self.linear_a.weight)
        nn.init.xavier_uniform_(self.linear_v.weight)
        nn.init.xavier_uniform_(self.speaker_weights_l1.weight)
        nn.init.xavier_uniform_(self.speaker_weights_a1.weight)
        nn.init.xavier_uniform_(self.speaker_weights_v1.weight)
        nn.init.xavier_uniform_(self.speaker_weights_l2.weight)
        nn.init.xavier_uniform_(self.speaker_weights_a2.weight)
        nn.init.xavier_uniform_(self.speaker_weights_v2.weight)
        # TODO
        if self.wo_persona == False:
            nn.init.xavier_uniform_(self.persona_transformation_l1.weight)
            nn.init.xavier_uniform_(self.persona_transformation_a1.weight)
            nn.init.xavier_uniform_(self.persona_transformation_v1.weight)
            nn.init.xavier_uniform_(self.persona_transformation_l2.weight)
            nn.init.xavier_uniform_(self.persona_transformation_a2.weight)
            nn.init.xavier_uniform_(self.persona_transformation_v2.weight)
        nn.init.xavier_uniform_(self.interactive_ES_transformation1.weight)
        nn.init.xavier_uniform_(self.interactive_ES_transformation2.weight)
        nn.init.xavier_uniform_(self.interactive_ERC_transformation1.weight)
        nn.init.xavier_uniform_(self.interactive_ERC_transformation2.weight)
        nn.init.xavier_uniform_(self.smax_fc_ERC.weight)
        nn.init.xavier_uniform_(self.smax_fc_ES.weight)



    


    def constructPersonaEdge(self, qmask, lengths):
      
        persona_edge_nums = 200000
        edge_index = torch.zeros([2, persona_edge_nums], dtype=torch.long)
      
        edge_count = 0
        uttr_bias = 0
        for dia_len in lengths:
         
            cur_dia_qmask = qmask[uttr_bias: uttr_bias + dia_len]
           
            cur_dia_speakers = sorted(list(set(torch.nonzero(cur_dia_qmask)[:, 1].tolist())))

            cur_dia_uttr_speaker = torch.nonzero(cur_dia_qmask)[:, 1]
           
            speaker_past_say = {cur_dia_speaker:0 for cur_dia_speaker in cur_dia_speakers}
 
            for i in range(dia_len):
        
                uttr_speaker = int(cur_dia_uttr_speaker[i])
         
                for j in range(speaker_past_say[uttr_speaker], i):
                
                    edge_index[0][edge_count] = uttr_bias + j
                  
                    edge_index[1][edge_count] = uttr_bias + i
                    edge_count += 1
           
                speaker_past_say[uttr_speaker] = i
       
            uttr_bias += dia_len
        edge_index = edge_index[:,:edge_count]
        return edge_index.to(qmask.device)

    def constructErcEdges(self, text, audio, visul, qmask, lengths, erc_windows=1):
        erc_edge_nums = 200000
        modals = []
        modal_data = {}
        modal_bias = {}
     
        if text != None:
            modals.append('l')
            modal_data['l'] = text
            modal_bias['l'] = (len(modal_data.keys()) - 1) * sum(lengths)
        if audio != None:
            modals.append('a')
            modal_data['a'] = audio
            modal_bias['a'] = (len(modal_data.keys()) - 1) * sum(lengths)
        if visul != None:
            modals.append('v')
            modal_data['v'] = visul
            modal_bias['v'] = (len(modal_data.keys()) - 1) * sum(lengths)
     
        erc_features = torch.cat([modal_data[modal] for modal in modals], dim=0)
        erc_edge_index = torch.zeros([2, erc_edge_nums], dtype=torch.long)
        erc_edge_indice = torch.zeros([2, erc_edge_nums], dtype=torch.long)
        erc_edge_type = torch.zeros([erc_edge_nums], dtype=torch.long)
        erc_edge_dialog = torch.zeros([erc_edge_nums], dtype=torch.long)
        erc_edge_count = 0

        #################################
        # add utterance multimodal edges
        # edges count: (len(modals) * len(modals) - len(modals)) * uttr_count
 
        uttr_bias = 0

        for dia_index in range(len(lengths)):
            dia_len = lengths[dia_index]

            for uttr_index in range(dia_len):
       
                for modal1 in modals:
                    for modal2 in modals:
           
                        erc_edge_index[0][erc_edge_count] = modal_bias[modal1] + uttr_bias + uttr_index
                        erc_edge_index[1][erc_edge_count] = modal_bias[modal2] + uttr_bias + uttr_index

                        erc_edge_indice[0][erc_edge_count] = uttr_index
                        erc_edge_indice[1][erc_edge_count] = uttr_index
            
                        erc_edge_dialog[erc_edge_count] = dia_index
                 
                        erc_edge_type[erc_edge_count] = 0
                        erc_edge_count = erc_edge_count + 1
      
            uttr_bias += dia_len

        ##################################
        # add neighbours edges

        uttr_bias = 0
        for dia_index in range(len(lengths)):
            dia_len = lengths[dia_index]
      
            cur_dia_qmask = qmask[uttr_bias: uttr_bias + dia_len]
           
            cur_dia_speakers = sorted(list(set(torch.nonzero(cur_dia_qmask)[:, 1].tolist())))
            cur_dia_uttr_speaker = torch.nonzero(cur_dia_qmask)[:, 1]
      
            speaker_past_say = {cur_dia_speaker: [0] for cur_dia_speaker in cur_dia_speakers}
            for i in range(dia_len):
                uttr_speaker = int(cur_dia_uttr_speaker[i])
                #################################
                # process intra and inter dependency
                for j in range(speaker_past_say[uttr_speaker][-erc_windows] if len(
                        speaker_past_say[uttr_speaker]) >= erc_windows else speaker_past_say[uttr_speaker][0], i):
                    # intra-speaker
                    if cur_dia_uttr_speaker[j] == cur_dia_uttr_speaker[i]:
             
                        for modal in modals:
                            erc_edge_index[0][erc_edge_count] = uttr_bias + j + modal_bias[modal]
                            erc_edge_index[1][erc_edge_count] = uttr_bias + i + modal_bias[modal]
                            erc_edge_indice[0][erc_edge_count] = j
                            erc_edge_indice[1][erc_edge_count] = i
                            # type = 1 mean intra edge
                            erc_edge_type[erc_edge_count] = 1
                            erc_edge_dialog[erc_edge_count] = dia_index
                            erc_edge_count += 1
          
                    else:
                        for modal in modals:
                            erc_edge_index[0][erc_edge_count] = uttr_bias + j + modal_bias[modal]
                            erc_edge_index[1][erc_edge_count] = uttr_bias + i + modal_bias[modal]
                            erc_edge_indice[0][erc_edge_count] = j
                            erc_edge_indice[1][erc_edge_count] = i
                            # type = 2 mean inter edge
                            erc_edge_type[erc_edge_count] = 2
                            erc_edge_dialog[erc_edge_count] = dia_index
                            erc_edge_count += 1
                speaker_past_say[uttr_speaker].append(i)
            uttr_bias += dia_len

        # assert erc_edge_count == erc_edge_amount
        erc_edge_index = erc_edge_index[:, :erc_edge_count]
        erc_edge_indice = erc_edge_indice[:, :erc_edge_count]
        erc_edge_type = erc_edge_type[:erc_edge_count]
        erc_edge_dialog = erc_edge_dialog[:erc_edge_count]
        return erc_features, erc_edge_index.to(qmask.device), erc_edge_indice.to(qmask.device), erc_edge_type.to(
            qmask.device), erc_edge_dialog.to(qmask.device), modals, modal_bias

    def getSemanticsSimilar(self,erc_features, combinations_result):


  
        cosine_similarities = []
        for pair in combinations_result:
            start_idx, end_idx = pair  
            start_node = erc_features[start_idx]  
            end_node = erc_features[end_idx]  

      
            start_node = normalize(start_node, p=2, dim=-1)
            end_node = normalize(end_node, p=2, dim=-1)

            similarity = cosine_similarity(start_node.unsqueeze(0), end_node.unsqueeze(0)).item()
            cosine_similarities.append(similarity)


        edges_with_similarities = list(zip(combinations_result, cosine_similarities))


        end_node_similarities = {}

        for (start_idx, end_idx), sim in edges_with_similarities:
            if end_idx not in end_node_similarities:
                end_node_similarities[end_idx] = []
            end_node_similarities[end_idx].append(sim)

     
        average_similarities = {end_node: np.mean(similarities) for end_node, similarities in
                                end_node_similarities.items()}

        filtered_edge = []
        filtered_edges = []
        for (start_idx, end_idx), sim in edges_with_similarities:
            avg_sim = average_similarities[end_idx]  
            if sim >= avg_sim:  
                filtered_edges.append((start_idx, end_idx))
                filtered_edge.append(((start_idx, end_idx), sim))

        return filtered_edges

    def constructErcByShiftEdges(self, text, audio, visul, qmask, lengths, erc_windows=1, speaker_same_label_segments=None, utt_same_label_segments=None):
        erc_edge_nums = 20000000
        modals = []
        modal_data = {}
        modal_bias = {}
        
        if text != None:
            modals.append('l')
            modal_data['l'] = text
            modal_bias['l'] = (len(modal_data.keys()) - 1) * sum(lengths)
        if audio != None:
            modals.append('a')
            modal_data['a'] = audio
            modal_bias['a'] = (len(modal_data.keys()) - 1) * sum(lengths)
        if visul != None:
            modals.append('v')
            modal_data['v'] = visul
            modal_bias['v'] = (len(modal_data.keys()) - 1) * sum(lengths)

        erc_features = torch.cat([modal_data[modal] for modal in modals], dim=0)
        erc_edge_index = torch.zeros([2, erc_edge_nums], dtype=torch.long)
        erc_edge_indice = torch.zeros([2, erc_edge_nums], dtype=torch.long)
        erc_edge_type = torch.zeros([erc_edge_nums], dtype=torch.long)
        erc_edge_dialog = torch.zeros([erc_edge_nums], dtype=torch.long)
        erc_edge_count = 0

        #################################
        # add utterance multimodal edges
        # edges count: (len(modals) * len(modals) - len(modals)) * uttr_count

        uttr_bias = 0
        for dia_index in range(len(lengths)):
            dia_len = lengths[dia_index]

            for uttr_index in range(dia_len):
       
                for modal1 in modals:
                    for modal2 in modals:

                        erc_edge_index[0][erc_edge_count] = modal_bias[modal1] + uttr_bias + uttr_index
                        erc_edge_index[1][erc_edge_count] = modal_bias[modal2] + uttr_bias + uttr_index

                        erc_edge_indice[0][erc_edge_count] = uttr_index
                        erc_edge_indice[1][erc_edge_count] = uttr_index
   
                        erc_edge_dialog[erc_edge_count] = dia_index
  
                        erc_edge_type[erc_edge_count] = 0
                        erc_edge_count = erc_edge_count + 1

            uttr_bias += dia_len

        ##################################
        # add neighbours edges

        erc_edge_true1 = []
        erc_edge_true2 = []
        erc_edge_true3 = []
        erc_edge_true4 = []

        uttr_bias = 0
        for dia_index in range(len(lengths)):
            label_segments_length = {}
            dia_len = lengths[dia_index]

            cur_dia_qmask = qmask[uttr_bias: uttr_bias + dia_len]

            cur_dia_uttr_speaker = torch.nonzero(cur_dia_qmask)[:, 1]


            dia_speaker = sorted(set(torch.nonzero(qmask[dia_index: dia_index + dia_len])[:, 1].tolist()))

            for uttr_speaker in dia_speaker:

                speaker_index = torch.nonzero(qmask[uttr_bias: uttr_bias + dia_len])[:, 1] == uttr_speaker
                local_indices = torch.arange(dia_len, device=qmask.device)[speaker_index.bool()].tolist()

                speaker_windows = speaker_same_label_segments.get(dia_index, {}).get(uttr_speaker, [])

                # label_segments_length[uttr_speaker] = len(speaker_windows)
                #################################
                # process intra and inter dependency
                for start, end in speaker_windows:
                    if start == end:
                        j = local_indices.index(end)
                        start = max(0, j - erc_windows)
                        start = local_indices[start]
                        end = min(end, dia_len - 1)

                        filtered_lists = local_indices[local_indices.index(start):local_indices.index(end) + 1]

                        if len(filtered_lists) == 1:
                            combinations_result = [(filtered_lists[0], filtered_lists[0])]
                        else:
                            combinations_result = list(combinations(filtered_lists, 2))
                        combinations_result = self.getSemanticsSimilar(erc_features, combinations_result)
                    else:

                        start = max(start, 0)
                        end = min(end, dia_len - 1)
                        filtered_lists = local_indices[local_indices.index(start):local_indices.index(end) + 1]
     
                        combinations_result = list(combinations(filtered_lists, 2))
                        combinations_result = self.getSemanticsSimilar(erc_features, combinations_result)
                    for j,n in combinations_result:

        
                        for modal in modals:
                            erc_edge_index[0][erc_edge_count] = uttr_bias + j + modal_bias[modal]
                            erc_edge_index[1][erc_edge_count] = uttr_bias + n + modal_bias[modal]
                            erc_edge_indice[0][erc_edge_count] = j
                            erc_edge_indice[1][erc_edge_count] = n
                            # type = 1 mean intra edge
                            erc_edge_type[erc_edge_count] = 1
                            erc_edge_dialog[erc_edge_count] = dia_index
                            erc_edge_count += 1
            # max_label_segments_speaker = max(label_segments_length, key=label_segments_length.get)
            # # inter-speaker
            # speaker_windows = speaker_same_label_segments.get(dia_index, {}).get(max_label_segments_speaker, [])

            utt_windows = utt_same_label_segments.get(dia_index, {})
            for start, end in utt_windows:
  
                if start == end:
     
                    start = max(0, start - erc_windows)
                    end = min(end, dia_len - 1)
                    filtered_lists = list(range(start, end + 1))
       
                    if len(filtered_lists) == 1:
                        combinations_result = [(filtered_lists[0], filtered_lists[0])]
                    else:
                        combinations_result = list(combinations(filtered_lists, 2))

                    combinations_result = self.getSemanticsSimilar(erc_features, combinations_result)
                else:
           
                    start = max(start, 0)
                    end = min(end, dia_len - 1)
                    filtered_lists = list(range(start, end + 1))
        
                    combinations_result = list(combinations(filtered_lists, 2))
                    combinations_result = self.getSemanticsSimilar(erc_features, combinations_result)


                for j, n in combinations_result:
                    for modal in modals:

                        erc_edge_index[0][erc_edge_count] = uttr_bias + j + modal_bias[modal]
                        erc_edge_index[1][erc_edge_count] = uttr_bias + n + modal_bias[modal]
                        erc_edge_indice[0][erc_edge_count] = j
                        erc_edge_indice[1][erc_edge_count] = n
                        # erc_edge_true3.append(j)
                        # erc_edge_true4.append(n)
                        # type = 2 mean inter edge
                        erc_edge_type[erc_edge_count] = 2
                        erc_edge_dialog[erc_edge_count] = dia_index
                        erc_edge_count += 1

            uttr_bias += dia_len
        # assert erc_edge_count == erc_edge_amount
        erc_edge_index = erc_edge_index[:, :erc_edge_count]
        erc_edge_indice = erc_edge_indice[:, :erc_edge_count]
        erc_edge_type = erc_edge_type[:erc_edge_count]
        erc_edge_dialog = erc_edge_dialog[:erc_edge_count]
        return erc_features, erc_edge_index.to(qmask.device), erc_edge_indice.to(qmask.device), erc_edge_type.to(
            qmask.device), erc_edge_dialog.to(qmask.device), modals, modal_bias


    def constructErcEdges(self, text, audio, visul, qmask, lengths, erc_windows=1):
        erc_edge_nums = 200000
        modals = []
        modal_data = {}
        modal_bias = {}
        if text != None:
            modals.append('l')
            modal_data['l'] = text
            modal_bias['l'] = (len(modal_data.keys()) - 1) * sum(lengths)
        if audio != None:
            modals.append('a')
            modal_data['a'] = audio
            modal_bias['a'] = (len(modal_data.keys()) - 1) * sum(lengths)
        if visul != None:
            modals.append('v')
            modal_data['v'] = visul
            modal_bias['v'] = (len(modal_data.keys()) - 1) * sum(lengths)

        erc_features = torch.cat([modal_data[modal] for modal in modals], dim=0)
        erc_edge_index = torch.zeros([2, erc_edge_nums], dtype=torch.long)
        erc_edge_indice = torch.zeros([2, erc_edge_nums], dtype=torch.long)
        erc_edge_type = torch.zeros([erc_edge_nums], dtype=torch.long)
        erc_edge_dialog = torch.zeros([erc_edge_nums], dtype=torch.long)
        erc_edge_count = 0

        #################################
        # add utterance multimodal edges
        # edges count: (len(modals) * len(modals) - len(modals)) * uttr_count
        uttr_bias = 0
        for dia_index in range(len(lengths)):
            dia_len = lengths[dia_index]
    
            for uttr_index in range(dia_len):
          
                for modal1 in modals:
                    for modal2 in modals:
          
                        erc_edge_index[0][erc_edge_count] = modal_bias[modal1] + uttr_bias + uttr_index
                        erc_edge_index[1][erc_edge_count] = modal_bias[modal2] + uttr_bias + uttr_index
            
                        erc_edge_indice[0][erc_edge_count] = uttr_index
                        erc_edge_indice[1][erc_edge_count] = uttr_index
             
                        erc_edge_dialog[erc_edge_count] = dia_index
              
                        erc_edge_type[erc_edge_count] = 0
                        erc_edge_count = erc_edge_count + 1
       
            uttr_bias += dia_len
                
        ##################################
        # add neighbours edges
        uttr_bias = 0
        for dia_index in range(len(lengths)):
            dia_len = lengths[dia_index]
            cur_dia_qmask = qmask[uttr_bias: uttr_bias + dia_len]
            
            cur_dia_speakers = sorted(list(set(torch.nonzero(cur_dia_qmask)[:, 1].tolist())))
            cur_dia_uttr_speaker = torch.nonzero(cur_dia_qmask)[:, 1]
            speaker_past_say = {cur_dia_speaker:[0] for cur_dia_speaker in cur_dia_speakers}
            for i in range(dia_len):
                uttr_speaker = int(cur_dia_uttr_speaker[i])
                #################################
                # process intra and inter dependency
                for j in range(speaker_past_say[uttr_speaker][-erc_windows] if len(speaker_past_say[uttr_speaker]) >= erc_windows else speaker_past_say[uttr_speaker][0], i):
                    # intra-speaker
                    if cur_dia_uttr_speaker[j] == cur_dia_uttr_speaker[i]:
                        for modal in modals:
                            erc_edge_index[0][erc_edge_count] = uttr_bias + j + modal_bias[modal]
                            erc_edge_index[1][erc_edge_count] = uttr_bias + i + modal_bias[modal]
                            erc_edge_indice[0][erc_edge_count] = j
                            erc_edge_indice[1][erc_edge_count] = i
                            # type = 1 mean intra edge
                            erc_edge_type[erc_edge_count] = 1
                            erc_edge_dialog[erc_edge_count] = dia_index
                            erc_edge_count += 1
                    # inter-speaker
                    else:
                        for modal in modals:
                            erc_edge_index[0][erc_edge_count] = uttr_bias + j + modal_bias[modal]
                            erc_edge_index[1][erc_edge_count] = uttr_bias + i + modal_bias[modal]
                            erc_edge_indice[0][erc_edge_count] = j
                            erc_edge_indice[1][erc_edge_count] = i
                            # type = 2 mean inter edge
                            erc_edge_type[erc_edge_count] = 2
                            erc_edge_dialog[erc_edge_count] = dia_index
                            erc_edge_count += 1
                speaker_past_say[uttr_speaker].append(i)
            uttr_bias += dia_len
                        
        # assert erc_edge_count == erc_edge_amount
        erc_edge_index = erc_edge_index[:,:erc_edge_count]
        erc_edge_indice = erc_edge_indice[:,:erc_edge_count]
        erc_edge_type = erc_edge_type[:erc_edge_count]
        erc_edge_dialog = erc_edge_dialog[:erc_edge_count]
        return erc_features, erc_edge_index.to(qmask.device), erc_edge_indice.to(qmask.device), erc_edge_type.to(qmask.device), erc_edge_dialog.to(qmask.device), modals, modal_bias

    def constructShiftEdges(self, text, audio, visul, qmask, lengths, shift_windows=1, edge_index_bias=0):
        shift_edge_nums = 200000
        modals = []
        modal_data = {}
        modal_bias = {}
        if text != None:
            modals.append('l')
            modal_data['l'] = text
            modal_bias['l'] = (len(modal_data.keys()) - 1) * sum(lengths)
        if audio != None:
            modals.append('a')
            modal_data['a'] = audio
            modal_bias['a'] = (len(modal_data.keys()) - 1) * sum(lengths)
        if visul != None:
            modals.append('v')
            modal_data['v'] = visul
            modal_bias['v'] = (len(modal_data.keys()) - 1) * sum(lengths)
        
        shift_features = torch.cat([modal_data[modal] for modal in modals], dim=0)
        
        shift_edge_index = torch.zeros([2, shift_edge_nums], dtype=torch.long)
        shift_edge_indice = torch.zeros([2, shift_edge_nums], dtype=torch.long)
        shift_edge_type = torch.zeros([shift_edge_nums], dtype=torch.long)
        shift_edge_dialog = torch.zeros([shift_edge_nums], dtype=torch.long)
        shift_edge_count = 0

        #################################
        # add utterance multimodal edges
        # edges count: (len(modals) * len(modals) - len(modals)) * uttr_count
        uttr_bias = 0
        for dia_index in range(len(lengths)):
            dia_len = lengths[dia_index]
            for uttr_index in range(dia_len):
                for modal1 in modals:
                    for modal2 in modals:
                        shift_edge_index[0][shift_edge_count] = modal_bias[modal1] + uttr_bias + uttr_index + edge_index_bias
                        shift_edge_index[1][shift_edge_count] = modal_bias[modal2] + uttr_bias + uttr_index + edge_index_bias
                        shift_edge_indice[0][shift_edge_count] = uttr_index
                        shift_edge_indice[1][shift_edge_count] = uttr_index
                        shift_edge_type[shift_edge_count] = 3
                        shift_edge_dialog[shift_edge_count] = dia_index
                        shift_edge_count = shift_edge_count + 1
            uttr_bias += dia_len


        #################################
        # add shift dependency edges
        # edges count:
        uttr_bias = 0
        for dia_index in range(len(lengths)):
            dia_len = lengths[dia_index]
            cur_dia_qmask = qmask[uttr_bias: uttr_bias + dia_len]
            cur_dia_speakers = sorted(list(set(torch.nonzero(cur_dia_qmask)[:, 1].tolist())))
            cur_dia_uttr_speaker = torch.nonzero(cur_dia_qmask)[:, 1]
            speaker_past_say = {cur_dia_speaker:[0] for cur_dia_speaker in cur_dia_speakers}
            for i in range(dia_len):
                uttr_speaker = int(cur_dia_uttr_speaker[i])
                #################################
                # process intra and inter dependency
                for j in range(speaker_past_say[uttr_speaker][-shift_windows] if len(speaker_past_say[uttr_speaker]) > shift_windows else speaker_past_say[uttr_speaker][0], i):
                    if cur_dia_uttr_speaker[j] == cur_dia_uttr_speaker[i]:
                        for modal in modals:
                            shift_edge_index[0][shift_edge_count] = uttr_bias + j + modal_bias[modal] + edge_index_bias
                            shift_edge_index[1][shift_edge_count] = uttr_bias + i + modal_bias[modal] + edge_index_bias
                            shift_edge_indice[0][shift_edge_count] = j
                            shift_edge_indice[1][shift_edge_count] = i
                            # type = 4 mean intra shift edge
                            shift_edge_type[shift_edge_count] = 4
                            shift_edge_dialog[shift_edge_count] = dia_index
                            shift_edge_count += 1
                    else:
                        for modal in modals:
                            shift_edge_index[0][shift_edge_count] = uttr_bias + j + modal_bias[modal] + edge_index_bias
                            shift_edge_index[1][shift_edge_count] = uttr_bias + i + modal_bias[modal] + edge_index_bias
                            shift_edge_indice[0][shift_edge_count] = j
                            shift_edge_indice[1][shift_edge_count] = i
                            # type = 5 mean inter shift edge
                            shift_edge_type[shift_edge_count] = 5
                            shift_edge_dialog[shift_edge_count] = dia_index
                            shift_edge_count += 1
                speaker_past_say[uttr_speaker].append(i)
            uttr_bias += dia_len
        shift_edge_index = shift_edge_index[:,:shift_edge_count]
        shift_edge_indice = shift_edge_indice[:,:shift_edge_count]
        shift_edge_type = shift_edge_type[:shift_edge_count]
        shift_edge_dialog = shift_edge_dialog[:shift_edge_count]
        return shift_features, shift_edge_index.to(qmask.device), shift_edge_indice.to(qmask.device), shift_edge_type.to(qmask.device), shift_edge_dialog.to(qmask.device), modals, modal_bias

    def constructInteractiveEdges(self, erc_features, shift_features, qmask, lengths, modals, modal_bias, interactive_windows = 1):
        task_length = sum(lengths) * len(modals)
        edge_num = 300000
        interactive_edge_index = torch.zeros([2, edge_num], dtype=torch.long)
        interactive_edge_indice = torch.zeros([2, edge_num], dtype=torch.long)
        interactive_edge_type = torch.zeros([edge_num], dtype=torch.long)
        interactive_edge_dialog = torch.zeros([edge_num], dtype=torch.long)
        interactive_edge_count = 0
        interactive_features = torch.cat([erc_features, shift_features], dim=0)
        #################################
        # add utterance multi task edges

        uttr_bias = 0
        for dia_index in range(len(lengths)):
            dia_len = lengths[dia_index]
            for uttr_index in range(dia_len):
                for neighbour_index in range(-interactive_windows, 1):
                    if uttr_index + neighbour_index < 0 or uttr_index + neighbour_index >=dia_len:
                        continue
                    else:
                        for modal_interactive in modals:
                            # erc -> shift
                            interactive_edge_index[0][interactive_edge_count] = uttr_bias + uttr_index + neighbour_index + modal_bias[modal_interactive]
                            interactive_edge_index[1][interactive_edge_count] = uttr_bias + uttr_index + modal_bias[modal_interactive] + task_length
                            interactive_edge_indice[0][interactive_edge_count] = uttr_index + neighbour_index
                            interactive_edge_indice[1][interactive_edge_count] = uttr_index
                            interactive_edge_type[interactive_edge_count] = 6
                            interactive_edge_dialog[interactive_edge_count] = dia_index
                            interactive_edge_count += 1

                            # shift -> erc
                            interactive_edge_index[0][interactive_edge_count] = uttr_bias + uttr_index + neighbour_index + modal_bias[modal_interactive] + task_length
                            interactive_edge_index[1][interactive_edge_count] = uttr_bias + uttr_index + modal_bias[modal_interactive]
                            interactive_edge_indice[0][interactive_edge_count] = uttr_index + neighbour_index
                            interactive_edge_indice[1][interactive_edge_count] = uttr_index
                            interactive_edge_type[interactive_edge_count] = 7
                            interactive_edge_dialog[interactive_edge_count] = dia_index
                            interactive_edge_count += 1
            uttr_bias += dia_len
        interactive_edge_index = interactive_edge_index[:,:interactive_edge_count]
        interactive_edge_indice = interactive_edge_indice[:,:interactive_edge_count]
        interactive_edge_type = interactive_edge_type[:interactive_edge_count]
        interactive_edge_dialog = interactive_edge_dialog[:interactive_edge_count]
        return interactive_features, interactive_edge_index.to(erc_features.device), interactive_edge_indice.to(erc_features.device), interactive_edge_type.to(erc_features.device), interactive_edge_dialog.to(erc_features.device)



    def constructShiftData(self, shift_features, qmask, lengths, node_bias, node_modals):
        features = {}
        for modal in node_modals:
            features[modal] = shift_features[node_bias[modal]: node_bias[modal] + sum(lengths)]
        person_shift_data = {modal: [] for modal in node_modals}
        uttr_count = 0
        for dia_len in lengths:
            dia_speaker = sorted(set(torch.nonzero(qmask[uttr_count: uttr_count + dia_len])[:, 1].tolist()))
            for speaker in dia_speaker:
                speaker_index = torch.nonzero(qmask[uttr_count: uttr_count + dia_len])[:, 1] == speaker
                for modal in node_modals:
                    current_dia_speaker_features = features[modal][uttr_count: uttr_count + dia_len][
                        speaker_index.bool()]
                    if current_dia_speaker_features.shape[0] > 1:
                        person_shift_data[modal].append(
                            torch.cat([current_dia_speaker_features[:-1], current_dia_speaker_features[1:]], dim=-1))
            uttr_count = uttr_count + dia_len

        context_shift_data = {modal: [] for modal in node_modals}
        uttr_count = 0
        for dia_len in lengths:
            for modal in node_modals:
                current_dia_features = features[modal][uttr_count: uttr_count + dia_len]
                if current_dia_features.shape[0] > 1:
                    context_shift_data[modal].append(
                        torch.cat([current_dia_features[:-1], current_dia_features[1:]], dim=-1))
            uttr_count = uttr_count + dia_len

        for modal in node_modals:
            person_shift_data[modal] = torch.cat(person_shift_data[modal], dim=0)
            context_shift_data[modal] = torch.cat(context_shift_data[modal], dim=0)

        return person_shift_data, context_shift_data

    def decodeShiftLabel(self, shift_label, dataset):
        if dataset == 'IEMOCAP':
            class_num = 6
        else:
            class_num = 7
        front_label = [int(x // class_num) for x in shift_label]
        back_label = [int(x % class_num) for x in shift_label]
        return front_label, back_label

    def constructErcEdgeByShiftLabel(self, qmask, lengths, speaker_label, context_label, class_num):
        # front_label, back_label = self.decodeShiftLabel(label[0], self.dataset)

        person_same_label_segments = {}
        label_list = speaker_label[0]
        speaker_uttr_count = 0
        cur = 0
        for idx, dia_len in enumerate(lengths):

            same_label_segs = {}

            dia_speaker = sorted(
                set(torch.nonzero(qmask[speaker_uttr_count: speaker_uttr_count + dia_len])[:, 1].tolist()))

            for speaker in dia_speaker:
                same_label = []
                speaker_index = torch.nonzero(qmask[speaker_uttr_count: speaker_uttr_count + dia_len])[:, 1] == speaker
                local_indices = torch.arange(dia_len, device=qmask.device)[speaker_index.bool()]
                start_idx = 0

                if local_indices.shape[0] > 0:
                    for i in range(0, local_indices.shape[0] - 1):

                        if label_list[cur] % (class_num + 1) != 0:
                            same_label.append((local_indices[start_idx].item(), local_indices[i].item()))
                            start_idx = i + 1
                        cur = cur + 1
                    same_label.append((local_indices[start_idx].item(), local_indices[-1].item()))
                else:
                    same_label.append((local_indices[start_idx].item(), local_indices[-1].item()))

                same_label_segs[speaker] = same_label
                person_same_label_segments[idx] = same_label_segs
            speaker_uttr_count = speaker_uttr_count + dia_len

        utt_same_label_segments = {}
        cur = 0
        label_list = context_label[0]
        for idx, dia_len in enumerate(lengths):
            same_label = []
            current_labels = label_list[cur: cur + dia_len - 1]
            start_idx = 0

            for i in range(len(current_labels)):
                if current_labels[i] % (class_num + 1) != 0:
                    same_label.append((start_idx, i))
                    start_idx = i + 1
                cur = cur + 1
            same_label.append((start_idx, dia_len - 1))

            utt_same_label_segments[idx] = same_label

        return person_same_label_segments, utt_same_label_segments

    def constructErcByShiftLabel(self, qmask, lengths, label, class_num):
        front_label, back_label = self.decodeShiftLabel(label[0], self.dataset)

        shift_label = []
        speaker_uttr_count = 0
        speaker_same_label_segments = {}

        for idx, dia_len in enumerate(lengths):

            same_label_segs = {}

            dia_speaker = sorted(
                set(torch.nonzero(qmask[speaker_uttr_count: speaker_uttr_count + dia_len])[:, 1].tolist()))

            for speaker in dia_speaker:
                same_label = []

                speaker_index = torch.nonzero(qmask[speaker_uttr_count: speaker_uttr_count + dia_len])[:, 1] == speaker
                current_speaker_label = label[speaker_uttr_count: speaker_uttr_count + dia_len][speaker_index.bool()]

                local_indices = torch.arange(dia_len, device=qmask.device)[speaker_index.bool()]

                start_idx = 0
                if current_speaker_label.shape[0] > 1:

                    for i in range(current_speaker_label.shape[0] - 1):

                        if current_speaker_label[i] != current_speaker_label[i + 1]:
                            same_label.append((local_indices[start_idx].item(), local_indices[i].item()))
                            start_idx = i + 1

                    same_label.append((local_indices[start_idx].item(), local_indices[-1].item()))
                else:
                    same_label.append((local_indices[start_idx].item(), local_indices[-1].item()))
                same_label_segs[speaker] = same_label
                speaker_same_label_segments[idx] = same_label_segs

            speaker_uttr_count = speaker_uttr_count + dia_len
        utt_same_label_segments = {}
        utt_uttr_count = 0
        for idx, dia_len in enumerate(lengths):
            same_label = []
            current_labels = label[utt_uttr_count: utt_uttr_count + dia_len]
            start_idx = 0
            if current_labels.shape[0] > 1:
                for i in range(current_labels.shape[0] - 1):

                    if current_labels[i] != current_labels[i + 1]:
                        same_label.append((start_idx, i))
                        start_idx = i + 1

                same_label.append((start_idx, dia_len - 1))
            else:
                same_label.append((start_idx, dia_len - 1))
            utt_same_label_segments[idx] = same_label

            utt_uttr_count = utt_uttr_count + dia_len

        return speaker_same_label_segments, utt_same_label_segments

    def forward(self, U, qmask, umask, lengths, U_a=None, U_v=None, test_label=False, speaker=None, labels=None, persona=None, wo_ercByShiftEdges=False):
        [r1,r2,r3,r4]=U
        seq_len, _, feature_dim = r1.size()
        if self.norm_strategy == 'LN':
            r1 = self.normLNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r2 = self.normLNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r3 = self.normLNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r4 = self.normLNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        elif self.norm_strategy == 'BN':
            r1 = self.normBNa(r1.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r2 = self.normBNb(r2.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r3 = self.normBNc(r3.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
            r4 = self.normBNd(r4.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        else:
            pass
        U = (r1 + r2 + r3 + r4)/4
 
 
        U = self.dropout_forward(U)
        U_a = self.dropout_forward(U_a) if U_a is not None else None
        U_v = self.dropout_forward(U_v) if U_v is not None else None

        emotions_a, emotions_v, emotions_l = None, None, None
        if 'l' in self.modals:
            U = self.linear_l(U)
            emotions_l, hidden_l = self.lstm_l(U)
            U_, qmask_ = U.transpose(0, 1), qmask.transpose(0, 1)
            U_p_ = torch.zeros(U_.size()[0], U_.size()[1], emotions_l.shape[-1]).type(U.type())
            U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]  # default 2
            for b in range(U_.size(0)):
                for p in range(len(U_parties_)):
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                    if index_i.size(0) > 0:
                        U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]
            E_parties_ = [self.rnn_parties_l(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in range(len(U_parties_))]
            for b in range(U_p_.size(0)):
                for p in range(len(U_parties_)):
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                    if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
            U_p = U_p_.transpose(0, 1)
            emotions_l = torch.nn.Sigmoid()(self.speaker_weights_l1(emotions_l)) * emotions_l + torch.nn.Sigmoid()(self.speaker_weights_l2(U_p)) * U_p

        if 'a' in self.modals:
            U_a = self.linear_a(U_a)
            emotions_a = U_a
            if self.av_using_lstm:
                emotions_a, hidden_a = self.lstm_a(U_a)
            U_, qmask_ = U_a.transpose(0, 1), qmask.transpose(0, 1)
            U_p_ = torch.zeros(U_.size()[0], U_.size()[1], emotions_a.shape[-1]).type(U_a.type())
            U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]  # default 2
            for b in range(U_.size(0)):
                for p in range(len(U_parties_)):
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                    if index_i.size(0) > 0:
                        U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]
            E_parties_ = [self.rnn_parties_a(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in range(len(U_parties_))]
            for b in range(U_p_.size(0)):
                for p in range(len(U_parties_)):
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                    if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
            U_p = U_p_.transpose(0, 1)
            emotions_a = torch.nn.Sigmoid()(self.speaker_weights_a1(emotions_a)) * emotions_a + torch.nn.Sigmoid()(self.speaker_weights_a2(U_p)) * U_p
        if 'v' in self.modals:
            U_v = self.linear_v(U_v)
            emotions_v = U_v
            if self.av_using_lstm:
                emotions_v, hidden_v = self.lstm_v(U_v)
            U_, qmask_ = U_v.transpose(0, 1), qmask.transpose(0, 1)
            U_p_ = torch.zeros(U_.size()[0], U_.size()[1], emotions_v.shape[-1]).type(U_v.type())
            U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]  # default 2
            for b in range(U_.size(0)):
                for p in range(len(U_parties_)):
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                    if index_i.size(0) > 0:
                        U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]
            E_parties_ = [self.rnn_parties_v(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in range(len(U_parties_))]
            for b in range(U_p_.size(0)):
                for p in range(len(U_parties_)):
                    index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
                    if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
            U_p = U_p_.transpose(0, 1)
            emotions_v = torch.nn.Sigmoid()(self.speaker_weights_v1(emotions_v)) * emotions_v + torch.nn.Sigmoid()(self.speaker_weights_v2(U_p)) * U_p

        #####################################################
        ######## get features from batch data ###############
 
        features_l = self.simple_batch(emotions_l, lengths)
        features_a = self.simple_batch(emotions_a, lengths)
        features_v = self.simple_batch(emotions_v, lengths)
        if persona!=None:   
            persona = self.simple_batch(persona.transpose(0, 1), lengths)
        #####################################################


        #####################################################
        ######## get persona fusion ##########################
        qmask = torch.cat([qmask[:lengths[dia_len_index],dia_len_index,:] for dia_len_index in range(len(lengths))], dim=0)
        modal_length = features_l.shape[0] if features_l is not None else features_a.shape[0] if features_a is not None else features_v.shape[0]
        if self.wo_persona == False:
            if self.persona_transform:
                persona = self.linear_persona(persona)
            l_persona_edge = self.constructPersonaEdge(qmask=qmask, lengths=lengths) if features_l is not None else None
            a_persona_edge = self.constructPersonaEdge(qmask=qmask, lengths=lengths) if features_a is not None else None
            v_persona_edge = self.constructPersonaEdge(qmask=qmask, lengths=lengths) if features_v is not None else None
            features_l_persona = self.personaGAT_l(features_l, persona, l_persona_edge) if features_l is not None else None
            features_a_persona = self.personaGAT_a(features_a, persona, a_persona_edge) if features_a is not None else None
            features_v_persona = self.personaGAT_v(features_v, persona, v_persona_edge) if features_v is not None else None

            features_l = torch.nn.Sigmoid()(self.persona_transformation_l1(features_l))*features_l + torch.nn.Sigmoid()(self.persona_transformation_l2(features_l_persona))*features_l_persona if features_l is not None else None
            features_a = torch.nn.Sigmoid()(self.persona_transformation_a1(features_a))*features_a + torch.nn.Sigmoid()(self.persona_transformation_a2(features_a_persona))*features_a_persona if features_a is not None else None
            features_v = torch.nn.Sigmoid()(self.persona_transformation_v1(features_v))*features_v + torch.nn.Sigmoid()(self.persona_transformation_v2(features_v_persona))*features_v_persona if features_v is not None else None

        ###########################
        ### process shift task ###
        features_shift, edge_index_shift, edge_indice_shift, edge_type_shift, edge_dialog_shift, modals, modal_bias = self.constructShiftEdges(
            features_l, features_a, features_v, qmask, lengths, shift_windows=self.shift_windows, edge_index_bias=0)

        features_shift_after = self.interactive_ES_GAT(features_shift, edge_index_shift, edge_indice_shift, edge_type_shift,
                                             edge_dialog_shift)
        features_shift = (torch.nn.Sigmoid()(self.interactive_ES_transformation1(features_shift)) * features_shift +
                    torch.nn.Sigmoid()(self.interactive_ES_transformation2(features_shift_after)) * features_shift_after)

        features_shift.retain_grad()  


        person_emotions_feat_shift, context_emotions_feat_shift = self.constructShiftData(features_shift, qmask, lengths, modal_bias, modals)
        person_emotions_feat_shift = torch.cat([person_emotions_feat_shift[modal] for modal in modals], dim=-1)
        person_emotions_feat_shift = self.dropout_smax_shift(person_emotions_feat_shift)
        person_emotions_feat_shift = nn.ReLU()(person_emotions_feat_shift)
        person_log_prob_shift = self.smax_fc_ES(person_emotions_feat_shift)
        person_log_prob_shift = torch.nn.functional.log_softmax(person_log_prob_shift, -1)

        context_emotions_feat_shift = torch.cat([context_emotions_feat_shift[modal] for modal in modals], dim=-1)
        context_emotions_feat_shift = self.dropout_smax_shift(context_emotions_feat_shift)
        context_emotions_feat_shift = nn.ReLU()(context_emotions_feat_shift)
        context_log_prob_shift = self.smax_fc_ES(context_emotions_feat_shift)
        context_log_prob_shift = torch.nn.functional.log_softmax(context_log_prob_shift, -1)
        ###########################
        #####################################################
        ######## stage #################################

        ###########################
        ###  shift -> erc  ###
        person_shift_labels, person_shift_preds, context_shift_preds = [], [], []
        person_shift_preds.append(torch.argmax(person_log_prob_shift, 1).cpu().numpy())
        context_shift_preds.append(torch.argmax(context_log_prob_shift, 1).cpu().numpy())

        ###########################
        ###  process erc task  ###
        person_same_label_segments, utt_same_label_segments = self.constructErcEdgeByShiftLabel(qmask, lengths, person_shift_preds, context_shift_preds, self.class_num)
        if wo_ercByShiftEdges:
            features_erc, edge_index_erc, edge_indice_erc, edge_type_erc, edge_dialog_erc, modals, modal_bias = self.constructErcEdges(features_l, features_a, features_v, qmask, lengths, erc_windows=self.erc_windows)
        else:
            features_erc, edge_index_erc, edge_indice_erc, edge_type_erc, edge_dialog_erc, modals, modal_bias = self.constructErcByShiftEdges(features_l, features_a, features_v, qmask, lengths, erc_windows=self.erc_windows, speaker_same_label_segments=person_same_label_segments, utt_same_label_segments=utt_same_label_segments)

        features_erc_after = self.interactive_ERC_GAT(features_erc, edge_index_erc, edge_indice_erc,edge_type_erc, edge_dialog_erc)
        features_erc =  (torch.nn.Sigmoid()(self.interactive_ERC_transformation1(features_erc)) * features_erc +
                     torch.nn.Sigmoid()(self.interactive_ERC_transformation2(features_erc_after))*features_erc_after)

        features_erc = features_erc + features_shift

        features_erc.retain_grad()  



        emotions_feat = torch.cat([features_erc[modal_index*modal_length : (modal_index+1)*modal_length] for modal_index in range(len(modals))], dim=-1)


        emotions_feat = self.dropout_smax_erc(emotions_feat)
        emotions_feat = nn.ReLU()(emotions_feat)
        log_prob = self.smax_fc_ERC(emotions_feat)
        log_prob = torch.nn.functional.log_softmax(log_prob, -1)



        return log_prob, person_log_prob_shift, context_log_prob_shift, emotions_feat, person_emotions_feat_shift, context_emotions_feat_shift, features_shift, features_erc


    def simple_batch(self, features, lengths):
        """
        Process Mini Batch data
        """
        if features == None:
            return None
        node_features = []
        batch_size = features.shape[1]
        for j in range(batch_size):
            node_features.append(features[:lengths[j], j, :])
        node_features = torch.cat(node_features, dim=0)
        return node_features


def pad(tensor, length, no_cuda):
    if isinstance(tensor, Variable):
        var = tensor

        if length > var.size(0):
            if not no_cuda:



                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            if not no_cuda:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor

