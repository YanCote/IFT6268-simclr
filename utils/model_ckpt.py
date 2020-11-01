import os

def save_session(epoch_save_step, ckpter, iteration, session, output="./output"):
    # Save checkpoint
    if iteration % epoch_save_step == 0 :
        directory = os.path.abspath(output)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        ckpter.save(session, os.path.join(directory, "session_{0}.ckpt".format(iteration)))