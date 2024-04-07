import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(score,mean):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title("Training ..")
    plt.xlabel("Number of games")
    plt.ylabel("Score")
    plt.plot(score)
    plt.plot(mean)
    plt.ylim(ymin=0)
    plt.text(len(score) - 1,score[-1],str(score[-1]))
    plt.text(len(mean) - 1,mean[-1],str(mean[-1]))
    plt.show(block=False)
    plt.pause(.1)