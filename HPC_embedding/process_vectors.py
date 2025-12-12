import sys
import os
import argparse
import numpy as np
import pickle

path = '/project2/nmherrer_110/kbliu/kbliu/unikp/embedding/unikp'
sys.path.append(path)
import torch
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc

def smiles_to_vec(Smiles):
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    mask_index = 4
    vocab = WordVocab.load_vocab(f'{path}/vocab.pkl')
    
    def get_inputs(sm):
        seq_len = 220
        sm = sm.split()
        if len(sm)>218:
            print('SMILES is too long ({:d})'.format(len(sm)))
            sm = sm[:109]+sm[-109:]
        ids = [vocab.stoi.get(token, unk_index) for token in sm]
        ids = [sos_index] + ids + [eos_index]
        seg = [1]*len(ids)
        padding = [pad_index]*(seq_len - len(ids))
        ids.extend(padding), seg.extend(padding)
        return ids, seg
    
    def get_array(smiles):
        x_id, x_seg = [], []
        for sm in smiles:
            a,b = get_inputs(sm)
            x_id.append(a)
            x_seg.append(b)
        return torch.tensor(x_id), torch.tensor(x_seg)
    
    trfm = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    trfm.load_state_dict(torch.load(f'{path}/trfm_12_23000.pkl', map_location='cpu', weights_only=True))
    trfm = trfm.to(torch.device('cpu'))
    trfm.eval()
    x_split = [split(sm) for sm in Smiles]
    xid, xseg = get_array(x_split)
    X = trfm.encode(torch.t(xid))
    return X

def Seq_to_vec(Sequence):
    for i in range(len(Sequence)):
        if len(Sequence[i]) > 1000:
            Sequence[i] = Sequence[i][:500] + Sequence[i][-500:]
    sequences_Example = []
    for i in range(len(Sequence)):
        zj = ''
        for j in range(len(Sequence[i]) - 1):
            zj += Sequence[i][j] + ' '
        zj += Sequence[i][-1]
        sequences_Example.append(zj)
    
    tokenizer = T5Tokenizer.from_pretrained(f"{path}/prot_t5_xl_uniref50", do_lower_case=False, legacy=True)
    model = T5EncoderModel.from_pretrained(f"{path}/prot_t5_xl_uniref50")
    gc.collect()
    print(torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    features = []
    for i in range(len(sequences_Example)):
        print('For sequence ', str(i+1))
        sequences_Example_i = sequences_Example[i]
        sequences_Example_i = [re.sub(r"[UZOB]", "X", sequences_Example_i)]
        ids = tokenizer.batch_encode_plus(sequences_Example_i, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu().numpy()
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len - 1]
            features.append(seq_emd)
    features_normalize = np.zeros([len(features), len(features[0][0])], dtype=float)
    for i in range(len(features)):
        for k in range(len(features[0][0])):
            for j in range(len(features[i])):
                features_normalize[i][k] += features[i][j][k]
            features_normalize[i][k] /= len(features[i])
    return features_normalize

def main():
    parser = argparse.ArgumentParser(description='Process molecular vectors')
    parser.add_argument('--data_type', type=str, required=True, choices=['kcat', 'km'],
                        help='Type of data to process (kcat or km)')
    parser.add_argument('--feature_type', type=str, required=True, choices=['smiles', 'sequence'],
                        help='Type of feature to process (smiles or sequence)')
    parser.add_argument('--task_id', type=int, required=True,
                        help='Task ID from SLURM array')
    parser.add_argument('--total_tasks', type=int, required=True,
                        help='Total number of tasks')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    if args.feature_type == 'smiles':
        filename = f'{args.data_type}_smiles.txt'
    else:
        filename = f'{args.data_type}_sequences.txt'
    
    with open(filename, 'r') as f:
        data = [line.strip() for line in f if line.strip()]
    
    print(f"Total data points: {len(data)}")
    
    # Calculate chunk for this task
    chunk_size = len(data) // args.total_tasks
    remainder = len(data) % args.total_tasks
    
    start_idx = args.task_id * chunk_size + min(args.task_id, remainder)
    end_idx = start_idx + chunk_size + (1 if args.task_id < remainder else 0)
    
    data_chunk = data[start_idx:end_idx]
    
    print(f"Processing task {args.task_id}: indices {start_idx} to {end_idx} ({len(data_chunk)} items)")
    
    # Process data
    if args.feature_type == 'smiles':
        vectors = smiles_to_vec(data_chunk)
    else:
        vectors = Seq_to_vec(data_chunk)
    
    # Save results
    output_file = os.path.join(
        args.output_dir, 
        f'{args.data_type}_{args.feature_type}_vectors_task_{args.task_id}.pkl'
    )
    
    with open(output_file, 'wb') as f:
        pickle.dump({
            'vectors': vectors,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'task_id': args.task_id
        }, f)
    
    print(f"Saved results to {output_file}")

if __name__ == '__main__':
    main()
