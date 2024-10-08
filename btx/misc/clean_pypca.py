def fuse_results(path,tag,num_nodes):
    fused_data = {}
    all_data = []
    for id_current_node in range(num_nodes):
        tag = f"{tag}_node_{id_current_node}"
        filename_with_tag = f"{path}pypca_model_{tag}.h5"
        all_data.append(unpack_model_file(filename_with_tag))
    
    fused_data['exp'] = all_data[0]['exp']
    fused_data['run'] = all_data[0]['run']
    fused_data['num_runs'] = all_data[0]['num_runs']
    fused_data['num_images'] = all_data[0]['num_images']
    fused_data['det_type'] = all_data[0]['det_type']
    fused_data['start_offset'] = all_data[0]['start_offset']
    fused_data['S'] = np.concatenate([data['S'] for data in all_data], axis=0)
    fused_data['V'] = np.concatenate([data['V'] for data in all_data], axis=0)
    fused_data['mu'] = np.concatenate([data['mu'] for data in all_data], axis=0)
        
    return fused_data


def unpack_model_file(filename):
    """
    Reads PyPCA model information from h5 file and returns its contents

    Parameters
    ----------
    filename: str
        name of h5 file you want to unpack

    Returns
    -------
    data: dict
        A dictionary containing the extracted data from the h5 file.
    """
    data = {}
    with h5py.File(filename, 'r') as f:
        data['exp'] = str(np.asarray(f.get('exp')))[2:-1]
        data['run'] = int(np.asarray(f.get('run')))
        data['num_runs'] = int(np.asarray(f.get('num_runs')))
        data['num_images'] = int(np.asarray(f.get('num_images')))
        data['det_type'] = str(np.asarray(f.get('det_type')))[2:-1]
        data['start_offset'] = int(np.asarray(f.get('start_offset')))
        data['S'] = np.asarray(f.get('S'))
        data['V'] = np.asarray(f.get('V'))
        data['mu'] = np.asarray(f.get('mu'))

    return data

def write_fused_data(data, path, tag):
    filename_with_tag = f"{path}pypca_model_{tag}.h5"

    with h5py.File(filename_with_tag, 'w') as f:
        f.create_dataset('exp', data=data['exp'])
        f.create_dataset('run', data=data['run'])
        f.create_dataset('num_runs', data=data['num_runs'])
        f.create_dataset('num_images', data=data['num_images'])
        f.create_dataset('det_type', data=data['det_type'])
        f.create_dataset('start_offset', data=data['start_offset'])
        f.create_dataset('S', data=data['S'])
        f.create_dataset('V', data=data['V'])
        f.create_dataset('mu', data=data['mu'])

def delete_node_models(path, tag, num_nodes):
    for id_current_node in range(num_nodes):
        tag = f"{tag}_node_{id_current_node}"
        filename_with_tag = f"{path}pypca_model_{tag}.h5"
        os.remove(filename_with_tag)

def clean_pypca(path, tag, num_nodes):
    fused_data = fuse_results(path, tag, num_nodes)
    write_fused_data(fused_data, path, tag)
    delete_node_models(path, tag, num_nodes)

