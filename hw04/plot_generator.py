
import matplotlib.pyplot as plt
import numpy as np

# Jiho Choi (jihochoi@snu.ac.kr)

def generate_plot():

    angle = np.arange(0, np.pi*2, 0.05)
    r = 50 + np.random.normal(0, 2, angle.shape)

    x = r * np.cos(angle)
    y = r * np.sin(angle)

    plt.rcParams["figure.figsize"] = (4, 4)  # 400x400
    plt.axis('off')
    plt.scatter(x, y, s=10, facecolors='none', edgecolors='black')

    # plt.save("./out/{0}_{1}.png".format(image_file, str(num_pc)))
    plt.savefig('02_gen_circle_plot.png')


def main():
    generate_plot()


print("============================================")
print("     Principal Component Analysis (PCA)     ")
print("============================================")

print("\n==========="); print("   START"); print("===========\n")
if __name__ == '__main__':
    main()
print("\n==========="); print("   E N D"); print("===========\n")