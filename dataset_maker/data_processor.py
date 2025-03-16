"""
Unified data processor for NeuroLM datasets
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Process EEG/EOG datasets for NeuroLM')
    
    parser.add_argument('dataset', type=str, choices=['workload', 'hmc', 'sleep_edf', 'eog_d', 'eog_e', 'eog_f', 'locked_in'],
                        help='Dataset to process')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory containing the raw dataset')
                        
    parser.add_argument('--output', type=str, default='/d:/datasets/NeuroLM/processed',
                        help='Output directory for processed data')
    
    args = parser.parse_args()
    
    # Make sure the output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Process the selected dataset
    if args.dataset == 'workload':
        from prepare_workload import process_dataset
        process_dataset(args.input, os.path.join(args.output, 'EEGWorkload'))
    
    elif args.dataset == 'hmc':
        from prepare_HMC import process_dataset
        process_dataset(args.input, os.path.join(args.output, 'HMC'))
    
    elif args.dataset == 'sleep_edf':
        from prepare_sleep_edf import process_dataset
        process_dataset(args.input, os.path.join(args.output, 'SleepEDF'))
    
    elif args.dataset == 'eog_d':
        from prepare_eog_data import prepare_eog_dataset
        prepare_eog_dataset('D', args.input, os.path.join(args.output, 'EOG_D'))
    
    elif args.dataset == 'eog_e':
        from prepare_eog_data import prepare_eog_dataset
        prepare_eog_dataset('E', args.input, os.path.join(args.output, 'EOG_E'))
    
    elif args.dataset == 'eog_f':
        from prepare_eog_data import prepare_eog_dataset
        prepare_eog_dataset('F', args.input, os.path.join(args.output, 'EOG_F'))
    
    elif args.dataset == 'locked_in':
        from prepare_locked_in import process_dataset
        process_dataset(args.input, os.path.join(args.output, 'LockedIn'))
    
    print(f"Processing of dataset {args.dataset} complete.")

if __name__ == "__main__":
    main()
