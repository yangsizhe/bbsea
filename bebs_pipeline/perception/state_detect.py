from .clip_utils import *


STATE_DICT = {
    'drawer': ['open', 'closed'],
    'cupboard': ['open', 'closed'],
    'mailbox': ['open', 'closed'],
    'catapult': ['triggered', 'not triggered']
}

object_name = 'drawer' # or anything existing in the scene whose state should be detected
states = STATE_DICT[object_name]
state_feats = get_text_feats([f'the {object_name} is {state}' for state in states])

def get_object_state(object_name, image):
    if object_name in STATE_DICT:
        states = STATE_DICT[object_name]

        # rank based on current image
        img_feats = get_img_feats(image)
        # state_feats = get_text_feats(states)
        state_feats = get_text_feats([f'the {object_name} is {state}' for state in states])
        sorted_states, sorted_scores = get_nn_text(states, state_feats, img_feats)
        return sorted_states[0]
    return None

def get_gt_object_state(object_name, env):
    if object_name not in STATE_DICT:
        return None
    
    if object_name == 'drawer':
        if env.obs.state.object_states['drawer/'].joint_states['drawer/middle_drawer_slide'].current_value > 0.08:
            return 'open'
        else: 
            return 'closed'
    elif object_name == 'catapult':
        if env.obs.state.object_states['catapult/'].joint_states['catapult/catapult_hinge'].current_value > 0.2:
            return 'triggered'
        else: 
            return 'not triggered'   
    elif object_name == 'cupboard':
        pass
    elif object_name == 'mailbox':
        pass
    else:
        return None