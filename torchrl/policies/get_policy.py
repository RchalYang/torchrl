
def get_policies():
    if len(env.observation_space.shape) == 3:
        params['net']['base_type']=networks.CNNBase
        if params['env']['frame_stack']:    
            buffer_param = params['replay_buffer'] 
            efficient_buffer = replay_buffers.MemoryEfficientReplayBuffer(int(buffer_param['size']))
            params['general_setting']['replay_buffer'] = efficient_buffer
    else:
        params['net']['base_type']=networks.MLPBase