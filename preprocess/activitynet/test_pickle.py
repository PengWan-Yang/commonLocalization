import pickle

# open a file, where you stored the pickled data

if True:
    _pickle_file = 'train_data.pkl'
    file = open(_pickle_file, 'rb')
    # dump information to that file
    data = pickle.load(file)
    for k,_list in data.items():
        for ele in _list:
                ele['fg_name'] = ele['fg_name'].replace('/home/dataset/v1-3/train_val_frames_3/','dataset/activitynet13/train_val_frames_3/')
                ele['bg_name'] = ele['bg_name'].replace('/home/dataset/v1-3/train_val_frames_3/','dataset/activitynet13/train_val_frames_3/')
    pickle.dump( data, open( _pickle_file, "wb" ) )
    print("done")

_pickle_file = 'test_data.pkl'
file = open(_pickle_file, 'rb')
# dump information to that file
data = pickle.load(file)
for _list in data:
    for ele in _list:
            ele['fg_name'] = ele['fg_name'].replace('/home/dataset/v1-3/train_val_frames_3/','dataset/activitynet13/train_val_frames_3/')
            ele['bg_name'] = ele['bg_name'].replace('/home/dataset/v1-3/train_val_frames_3/','dataset/activitynet13/train_val_frames_3/')
pickle.dump( data, open( _pickle_file, "wb" ) )
print("done")