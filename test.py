import torch


def constructErcEdgeByShiftLabel(qmask, lengths, speaker_label, context_label, class_num):
    # front_label, back_label = self.decodeShiftLabel(label[0], self.dataset)

    # 记录结果：每段话中同一说话者连续相同情绪的信息
    person_same_label_segments = {}
    label_list = speaker_label[0]
    speaker_uttr_count = 0
    cur = 0
    for idx, dia_len in enumerate(lengths):

        same_label_segs = {}

        # 提取说话者索引
        dia_speaker = sorted(
            set(torch.nonzero(qmask[speaker_uttr_count: speaker_uttr_count + dia_len])[:, 1].tolist()))

        for speaker in dia_speaker:
            same_label = []
            # 对每个说话者，找到当前对话中属于该说话者的句子索引
            speaker_index = torch.nonzero(qmask[speaker_uttr_count: speaker_uttr_count + dia_len])[:, 1] == speaker
            local_indices = torch.arange(dia_len, device=qmask.device)[speaker_index.bool()]
            start_idx = 0

            # 构建连续句子的标签组合
            if local_indices.shape[0] > 0:
                # 遍历连续的标签对，
                for i in range(0, local_indices.shape[0 ] -1):
                    # 统计连续相同标签的段
                    # 检查当前句子的标签是否与下一个句子的标签不同

                    if label_list[cur] % (class_num +1) != 0:
                        # 添加 (起始索引, 终点索引)
                        same_label.append((local_indices[start_idx].item(), local_indices[i].item()))
                        # 更新新的起始点
                        start_idx = i + 1
                    cur = cur + 1
                # 添加最后一段
                same_label.append((local_indices[start_idx].item(), local_indices[-1].item()))
            else:
                same_label.append((local_indices[start_idx].item(), local_indices[-1].item()))

            same_label_segs[speaker] = same_label
            person_same_label_segments[idx] = same_label_segs
        speaker_uttr_count = speaker_uttr_count + dia_len

    # 不区分说话者的标签转移子对话
    utt_same_label_segments = {}
    cur = 0
    label_list = context_label[0]
    for idx, dia_len in enumerate(lengths):
        same_label = []
        current_labels = label_list[cur: cur + dia_len - 1]
        start_idx = 0
        # 构建连续句子的标签组合

        for i in range(len(current_labels)):
            # 检查当前句子的标签是否与下一个句子的标签不同
            if current_labels[i] % (class_num +1) != 0:
                # 添加 (起始索引, 终点索引)
                same_label.append((start_idx, i))
                # 更新新的起始点
                start_idx = i + 1
            cur = cur + 1
        # 添加最后一段
        same_label.append((start_idx, dia_len - 1))

        utt_same_label_segments[idx] = same_label

    return person_same_label_segments, utt_same_label_segments

def constructShiftData(shift_features, qmask, lengths, node_bias, node_modals):
    features = {}
    for modal in node_modals:
        features[modal] = shift_features[node_bias[modal]: node_bias[modal] + sum(lengths)]
    # 区分说话者的情绪标签转移
    person_shift_data ={modal:[] for modal in node_modals}
    uttr_count = 0
    for dia_len in lengths:
        dia_speaker = sorted(set(torch.nonzero(qmask[uttr_count: uttr_count+dia_len])[:,1].tolist()))
        for speaker in dia_speaker:
            speaker_index = torch.nonzero(qmask[uttr_count: uttr_count+dia_len])[:,1] == speaker
            for modal in node_modals:
              # 从 features 中提取出对应说话者在当前对话的特征（句子特征）
                current_dia_speaker_features = features[modal][uttr_count: uttr_count+dia_len][speaker_index.bool()]
                # 将说话者的句子特征拼接
                if current_dia_speaker_features.shape[0] > 1:
                    person_shift_data[modal].append(torch.cat([current_dia_speaker_features[:-1], current_dia_speaker_features[1:]], dim=-1))
        uttr_count = uttr_count + dia_len


    # 不区分说话者的情绪标签转移
    context_shift_data = {modal: [] for modal in node_modals}
    uttr_count = 0
    for dia_len in lengths:
        for modal in node_modals:
            # 获取当前对话的所有句子特征（上下文特征）
            current_dia_features = features[modal][uttr_count: uttr_count + dia_len]
            # 构造上下文情绪转移：拼接相邻句子的特征
            if current_dia_features.shape[0] > 1:
                context_shift_data[modal].append(
                    torch.cat([current_dia_features[:-1], current_dia_features[1:]], dim=-1))
        # 更新 `uttr_count`，指向下一个对话的起始位置
        uttr_count = uttr_count + dia_len

    # 拼接所有模态
    for modal in node_modals:
        person_shift_data[modal] = torch.cat(person_shift_data[modal], dim=0)
        context_shift_data[modal] = torch.cat(context_shift_data[modal], dim=0)

    return person_shift_data, context_shift_data

def constructShiftLabel(qmask, lengths, label, class_num):
    # 保存每个对话中连续句子的组合标签
    person_shift_label = []

    uttr_count = 0
    for dia_len in lengths:
        # 提取说话者索引
        dia_speaker = sorted(set(torch.nonzero(qmask[uttr_count: uttr_count+dia_len])[:,1].tolist()))
        for speaker in dia_speaker:
            # 对每个说话者，找到当前对话中属于该说话者的句子索引
            speaker_index = torch.nonzero(qmask[uttr_count: uttr_count+dia_len])[:,1] == speaker
            # 标签张量 label 中提取出当前说话者的句子标签
            current_speaker_label = label[uttr_count: uttr_count+dia_len][speaker_index.bool()]
            # 构建连续句子的标签组合
            if current_speaker_label.shape[0] > 1:
                # 遍历连续的标签对，将当前句子的标签和下一个句子的标签组合成一个新的标签
                for i in range(current_speaker_label.shape[0]-1):
                    person_shift_label.append(current_speaker_label[i] * class_num + current_speaker_label[i+1])
        uttr_count = uttr_count + dia_len

    # 不区分说话者的情绪标签转移
    context_shift_label = []
    uttr_count = 0
    for dia_len in lengths:
        # 获取当前对话的所有句子标签（上下文标签）
        current_label = label[uttr_count: uttr_count + dia_len]
        # 构造上下文情绪转移：拼接相邻句子的特征

        if current_label.shape[0] > 1:
            for i in range(current_label.shape[0] - 1):
                context_shift_label.append(current_label[i] * class_num + current_label[i+1])
        # 更新 `uttr_count`，指向下一个对话的起始位置
        uttr_count = uttr_count + dia_len

    return torch.tensor(person_shift_label).to(qmask.device), torch.tensor(context_shift_label).to(qmask.device)

def constructErcByShiftEdges(text, audio, visul, qmask, lengths, erc_windows=1, edge_index_bias=0, speaker_same_label_segments=None, utt_same_label_segments=None):
    erc_edge_nums = 200000
    modals = []
    modal_data = {}
    modal_bias = {}
    # 为某个模态计算一个偏移量，偏移量的大小基于其他模态的数量以及当前批次中的对话句子数
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
    # 拼接每个模态的特征
    erc_features = torch.cat([modal_data[modal] for modal in modals], dim=0)
    erc_edge_index = torch.zeros([2, erc_edge_nums], dtype=torch.long)
    erc_edge_indice = torch.zeros([2, erc_edge_nums], dtype=torch.long)
    erc_edge_type = torch.zeros([erc_edge_nums], dtype=torch.long)
    erc_edge_dialog = torch.zeros([erc_edge_nums], dtype=torch.long)
    erc_edge_count = 0

    #################################
    # add utterance multimodal edges
    # edges count: (len(modals) * len(modals) - len(modals)) * uttr_count
    # 为每个对话构建多模态边
    uttr_bias = 0
    for dia_index in range(len(lengths)):
        dia_len = lengths[dia_index]
        # 为每个句子构建边
        for uttr_index in range(dia_len):
            # 双重循环，表示在两个模态之间建立边
            for modal1 in modals:
                for modal2 in modals:
                    # 通过 modal_bias 和句子偏移量计算出不同模态下同一句话的索引，并为它们添加一条边
                    erc_edge_index[0][erc_edge_count] = modal_bias[modal1] + uttr_bias + uttr_index + edge_index_bias
                    erc_edge_index[1][erc_edge_count] = modal_bias[modal2] + uttr_bias + uttr_index + edge_index_bias
                    # 记录当前句子的索引（用于对话中的定位）
                    erc_edge_indice[0][erc_edge_count] = uttr_index
                    erc_edge_indice[1][erc_edge_count] = uttr_index
                    # 记录这条边所属的对话索引
                    erc_edge_dialog[erc_edge_count] = dia_index
                    # 设为 0，表示当前的边是“同一句话在不同模态间”的连接
                    erc_edge_type[erc_edge_count] = 0
                    erc_edge_count = erc_edge_count + 1
        # 更新 uttr_bias，以确保下一个对话的句子索引正确偏移
        uttr_bias += dia_len

    ##################################
    # add neighbours edges
    # 为每个对话的句子构建“邻居边”（neighbours edges），通过添加**同一说话者（intra-speaker）和不同说话者（inter-speaker）**之间的依赖关系
    uttr_bias = 0
    for dia_index in range(len(lengths)):
        label_segments_length = {}
        dia_len = lengths[dia_index]
        # 获取当前对话的qmask
        cur_dia_qmask = qmask[uttr_bias: uttr_bias + dia_len]

        cur_dia_uttr_speaker = torch.nonzero(cur_dia_qmask)[:, 1]

        # 提取说话者索引
        dia_speaker = sorted(set(torch.nonzero(qmask[dia_index: dia_index + dia_len])[:, 1].tolist()))

        for uttr_speaker in dia_speaker:

            # 获取当前对话当前说话者的窗口范围
            speaker_windows = speaker_same_label_segments.get(dia_index, {}).get(uttr_speaker, [])
            # label_segments_length[uttr_speaker] = len(speaker_windows)
            #################################
            # process intra and inter dependency
            for start, end in speaker_windows:
                # 确保窗口范围合法，避免越界
                start = max(start, 0)
                end = min(end, dia_len - 1)
                # # 筛选窗口内由指定说话者说的句子的索引
                # indices = [i for i in range(start, end + 1) if cur_dia_uttr_speaker[i] == uttr_speaker]
                # 遍历窗口中的句子
                for j in range(start, end + 1):
                    dynamics_count = 0
                    for n in range(start, end + 1):
                        # intra-speaker（同一说话者）
                        if dynamics_count > erc_windows:
                            break
                        if cur_dia_uttr_speaker[j] == cur_dia_uttr_speaker[n]:
                            # 多个模态
                            for modal in modals:
                                erc_edge_index[0][erc_edge_count] = uttr_bias + j + modal_bias[modal] + edge_index_bias
                                erc_edge_index[1][erc_edge_count] = uttr_bias + n + modal_bias[modal] + edge_index_bias
                                erc_edge_indice[0][erc_edge_count] = j
                                erc_edge_indice[1][erc_edge_count] = n
                                # type = 1 mean intra edge
                                erc_edge_type[erc_edge_count] = 1
                                erc_edge_dialog[erc_edge_count] = dia_index
                                erc_edge_count += 1
                            dynamics_count += 1

        # max_label_segments_speaker = max(label_segments_length, key=label_segments_length.get)
        # # inter-speaker（不同说话者）
        # speaker_windows = speaker_same_label_segments.get(dia_index, {}).get(max_label_segments_speaker, [])

        utt_windows = utt_same_label_segments.get(dia_index, {})
        for start, end in utt_windows:
            # 确保窗口范围合法，避免越界
            start = max(start, 0)
            end = min(end, dia_len - 1)

            # 遍历窗口中的句子
            for j in range(start, end + 1):
                dynamics_count = 0
                for n in range(start, end + 1):
                    if dynamics_count > erc_windows:
                        break
                    if cur_dia_uttr_speaker[j] != cur_dia_uttr_speaker[n] or start == end:
                        # 控制如果子对话只有一句话，则连接上一句话
                        if start == end and j != 0 and cur_dia_uttr_speaker[j - 1] != cur_dia_uttr_speaker[n]:
                            j = j - 1
                        # 多个模态
                        for modal in modals:
                            erc_edge_index[0][erc_edge_count] = uttr_bias + j + modal_bias[modal] + edge_index_bias
                            erc_edge_index[1][erc_edge_count] = uttr_bias + n + modal_bias[modal] + edge_index_bias
                            erc_edge_indice[0][erc_edge_count] = j
                            erc_edge_indice[1][erc_edge_count] = n
                            # type = 2 mean inter edge
                            erc_edge_type[erc_edge_count] = 2
                            erc_edge_dialog[erc_edge_count] = dia_index
                            erc_edge_count += 1
                    # else:
                    #     for modal in modals:
                    #         erc_edge_index[0][erc_edge_count] = uttr_bias + j + modal_bias[modal]
                    #         erc_edge_index[1][erc_edge_count] = uttr_bias + n + modal_bias[modal]
                    #         erc_edge_indice[0][erc_edge_count] = j
                    #         erc_edge_indice[1][erc_edge_count] = n
                    #         # type = 1 mean intra edge
                    #         erc_edge_type[erc_edge_count] = 1
                    #         erc_edge_dialog[erc_edge_count] = dia_index
                    #         erc_edge_count += 1
                    dynamics_count += 1
        uttr_bias += dia_len
    # assert erc_edge_count == erc_edge_amount
    erc_edge_index = erc_edge_index[:, :erc_edge_count]
    erc_edge_indice = erc_edge_indice[:, :erc_edge_count]
    erc_edge_type = erc_edge_type[:erc_edge_count]
    erc_edge_dialog = erc_edge_dialog[:erc_edge_count]
    return erc_features, erc_edge_index.to(qmask.device), erc_edge_indice.to(qmask.device), erc_edge_type.to(
        qmask.device), erc_edge_dialog.to(qmask.device), modals, modal_bias
def main():
    print(1)
    # 每段对话中有效句子的数量
    lengths = [5,3,1]
    qmask = torch.tensor([
        [1., 0.],
        [1., 0.],
        [1., 0.],
        [0., 1.],
        [1., 0.],
        [1., 0.],
        [0., 1.],
        [1., 0.],
        [0., 1.]
    ])
    features_shift = torch.full((27, 1), 0.5)  # 用 0.5 填充
    features_l = torch.full((9, 1), 0.5)  # 用 0.5 填充
    features_v = torch.full((9, 1), 0.5)  # 用 0.5 填充
    features_a = torch.full((9, 1), 0.5)  # 用 0.5 填充
    modals={'l','a','v'}
    modal_bias = {'l':9,'a':0,'v':18}

    person_shift_preds=[[7,7,8,6,6,28,29,29]]
    context_shift_preds=[[7,7,8,6,6,28,29,29]]


    # 将 label 中每段对话的有效标签提取出来，并通过 torch.cat 将所有有效标签拼接成一个张量
    label = torch.tensor([2, 2, 5, 3, 3, 6, 1, 4, 4])


    # 将相邻的两个句子标签组合成一个新标签，并存储在 shift_label 中
    person_label_shift, context_label_shift = constructShiftLabel(qmask, lengths, label, 6)

    person_shift_preds = [person_label_shift.tolist()]
    context_shift_preds = [context_label_shift.tolist()]

    person_emotions_feat_shift, context_emotions_feat_shift = constructShiftData(features_shift, qmask, lengths,
                                                                                      modal_bias, modals)
    person_same_label_segments, utt_same_label_segments = constructErcEdgeByShiftLabel(qmask, lengths,
                                                                                       person_shift_preds,
                                                                                       context_shift_preds,
                                                                                       6)


    features_erc, edge_index_erc, edge_indice_erc, edge_type_erc, edge_dialog_erc, modals, modal_bias = constructErcByShiftEdges(
    features_l, features_a, features_v, qmask, lengths, 3, edge_index_bias=0,
    speaker_same_label_segments=person_same_label_segments, utt_same_label_segments=utt_same_label_segments)

if __name__ == "__main__":
    main()




