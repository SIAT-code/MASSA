import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_train_dev_loss(epochs, train_loss, eval_loss, base_path, save_name):
    plt.plot(epochs, train_loss, '#3fc1fd', label='Training loss(%s)' % save_name)
    plt.plot(epochs, eval_loss, '#d09fff', label='Validation loss(%s)' % save_name)
    # plt.plot([330, 330], [0.9773016059994697-0.1, 1.0378530149936677+0.1], '#fd8989', label='Take the model parameters of the epoch')
    plt.title('Training and validation loss on %s' % save_name)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(base_path, save_name+'.jpg'))
    plt.cla()