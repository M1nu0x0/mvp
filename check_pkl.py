import pickle

pkl_list = [
    '/home/aisl/panoptic-toolbox/group_validation_cam4.pkl',
    '/home/aisl/panoptic-toolbox/group_validation_cam5.pkl',
    '/home/aisl/panoptic-toolbox/group_validation_cam5_sub.pkl',
    '/home/aisl/panoptic-toolbox/group_train_cam4.pkl',
    '/home/aisl/panoptic-toolbox/group_train_cam5.pkl'
]

for pkl in pkl_list:
    with open(pkl, 'rb') as f:
        data = pickle.load(f)
    
    print(pkl)
    print(data['sequence_list'])
    print(data['interval'])
    print(data['cam_list'])
    print()
