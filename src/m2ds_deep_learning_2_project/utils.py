import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from sklearn.utils import shuffle
from tqdm.auto import tqdm


def lire_alpha_digit_one(char):
    """Read one character from the binaryalphadigs.mat file."""
    binaryalphadigs = scipy.io.loadmat("data/binaryalphadigs.mat")

    class_labels = (
        np.array(binaryalphadigs["classlabels"][0].tolist()).flatten().tolist()
    )
    index = class_labels.index(char)
    data = binaryalphadigs["dat"][index]

    result_list = [matrix.flatten() for matrix in data]

    return np.array(result_list)


def lire_alpha_digit(*chars):
    """Read characters from the binaryalphadigs.mat file."""
    result_list = [lire_alpha_digit_one(char) for char in chars]

    return np.concatenate(result_list)


def lire_MNIST(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Read MNIST data.

    Parameters
    ----------
    split : str
        "train" or "test"

    Returns
    -------
    X: np.ndarray, size (n, 784)
    y: np.ndarray, size (n)
    """
    data = scipy.io.loadmat("data/mnist_all.mat")

    X_list = []
    y_list = []
    for i in range(10):
        X_i = data[split + str(i)]
        y_i = i * np.ones(X_i.shape[0])
        X_list.append(X_i)
        y_list.append(y_i)
    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    X = X >= 128
    X = np.array(X, dtype=int)
    y = np.array(y, dtype=int)

    return X, y


def plot_grid(X, image_size):
    """Plot a grid of images."""
    fig, axes = plt.subplots(2, 10, figsize=(10, 2.5))

    for ax in axes.ravel():
        index = random.randint(0, len(X) - 1)
        image = X[index].reshape(image_size)
        ax.imshow(image)
        ax.axis(False)

    plt.tight_layout()

    plt.show()


# --------------------------------- RBM model -------------------------------- #


class RBM:
    def __init__(self, p: int, q: int):
        """Init RBM model.

        Parameters
        ----------
        p : int
        q : int
        """
        self.p = p
        self.q = q
        self.a = np.zeros(p)
        self.b = np.zeros(q)
        self.W = np.random.normal(loc=0, scale=np.sqrt(0.01), size=(p, q))

    def entree_sortie(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray, size (n, p)
            input data

        Returns
        -------
        np.ndarray, size (n, q)
            output data
        """
        sortie = 1 / (1 + np.exp(-(X @ self.W + self.b)))

        return sortie

    def sortie_entree(self, H: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        H : np.ndarray, size (n, q)
            output data

        Returns
        -------
        np.ndarray, size (n, p)
            input data
        """
        entree = 1 / (1 + np.exp(-(H @ self.W.T + self.a)))

        return entree

    def calcul_softmax(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray, size (n, p)
            input data

        Returns
        -------
        np.ndarray, size (n, p)
            softmax
        """
        sortie = X @ self.W + self.b

        proba = np.exp(sortie) / np.sum(np.exp(sortie), axis=1, keepdims=True)

        return proba

    def train(
        self,
        X: np.ndarray,
        n_epochs: int,
        learning_rate: float,
        batch_size: int,
        plot: bool = True,
    ):
        """Train the RBM.

        Parameters
        ----------
        X : np.ndarray, size (n, p)
            input data
        n_epochs : int
        learning_rate : float
        batch_size : int
        plot : bool, optional
            by default True
        """
        error_history = []
        with tqdm(range(n_epochs)) as pbar:
            for _ in pbar:
                X_copy = X.copy()
                np.random.shuffle(X_copy)

                for batch in range(0, X.shape[0], batch_size):
                    X_batch = X_copy[batch : min(batch + batch_size, X.shape[0])]
                    tb = X_batch.shape[0]

                    v0 = X_batch
                    p_h_v0 = self.entree_sortie(v0)

                    # Sample according to a Bernoulli
                    h0 = (np.random.random((tb, self.q)) < p_h_v0) * 1
                    p_v1_h0 = self.sortie_entree(h0)
                    v1 = (np.random.random((tb, self.p)) < p_v1_h0) * 1
                    p_h_v1 = self.entree_sortie(v1)

                    # Calculate the gradient
                    grad_a = np.sum(v0 - v1, axis=0)  # size p
                    grad_b = np.sum(p_h_v0 - p_h_v1, axis=0)  # size q
                    grad_W = v0.T @ p_h_v0 - v1.T @ p_h_v1

                    # Update the parameters
                    self.a += (learning_rate / tb) * grad_a
                    self.b += (learning_rate / tb) * grad_b
                    self.W += (learning_rate / tb) * grad_W

                H = self.entree_sortie(X_copy)
                X_reconstruction = self.sortie_entree(H)

                error = np.mean((X_copy - X_reconstruction) ** 2)
                error_history.append(error)
                pbar.set_description(f"error {error:.4f} ")
                # print(f"Reconstruction error: {error:.4f}")
        if plot:
            plt.plot(error_history)
            plt.grid()
            plt.show()

        return error_history

    def generer_image(self, n_images: int, n_gibbs: int) -> list:
        """
        Parameters
        ----------
        n_images : int
        n_gibbs : int

        Returns
        -------
        list
            list of images
        """
        output = []
        for _ in range(n_images):
            v = (np.random.random(self.p) < 1 / 2) * 1
            for _ in range(n_gibbs):
                h = (np.random.random(self.q) < self.entree_sortie(v)) * 1
                v = (np.random.random(self.p) < self.sortie_entree(h)) * 1
            output.append(v)
        return output


# --------------------------------- DBN model -------------------------------- #


class DBN:
    def __init__(self, config_list: list):
        """Init DBN model.

        Parameters
        ----------
        config_list : list
            list of config for each RBM inside the DBN
        """
        self.RBM_list = [RBM(*config) for config in config_list]

    def train(
        self,
        X: np.ndarray,
        n_epochs: int,
        learning_rate: float,
        batch_size: int,
        plot: bool = True,
    ):
        """Train the DBN.

        Parameters
        ----------
        X : np.ndarray, size (n, p)
            input data
        n_epochs : int
        learning_rate : float
        batch_size : int
        plot : bool, optional
            by default True
        """
        X = X.copy()
        error_history = []
        for rbm in self.RBM_list:
            history = rbm.train(X, n_epochs, learning_rate, batch_size, plot=plot)
            X = rbm.entree_sortie(X)
            error_history.append(history)
        return error_history

    def generer_image(self, n_images: int, n_gibbs: int) -> list:
        """
        Parameters
        ----------
        n_images : int
        n_gibbs : int

        Returns
        -------
        list
            list of images
        """
        output = self.RBM_list[-1].generer_image(n_images, n_gibbs)

        for rbm in self.RBM_list[len(self.RBM_list) - 2 :: -1]:
            for i in range(len(output)):
                v = output[i]
                temp = rbm.sortie_entree(v)
                output[i] = (np.random.random(temp.shape[0]) < temp) * 1

        return output


# --------------------------------- DNN model -------------------------------- #


class DNN:
    def __init__(self, config_list):
        """Init DNN model.

        Parameters
        ----------
        config_list : list
            list of config for each RBM inside the DBN
            + config for the classification layer.
        """
        self.DBN = DBN(config_list[:-1])
        self.RBM_classif = RBM(*config_list[-1])

    def pretrain(
        self,
        X: np.ndarray,
        n_epochs: int,
        learning_rate: float,
        batch_size: int,
        plot: bool = True,
    ):
        """Pretrain the DNN.

        Parameters
        ----------
        X : np.ndarray, size (n, p)
            input data
        n_epochs : int
        learning_rate : float
        batch_size : int
        plot : bool, optional
            by default True
        """
        self.DBN.train(
            X=X,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            plot=plot,
        )

    def entree_sortie_reseau(self, X: np.ndarray) -> tuple:
        """
        Parameters
        ----------
        X : np.ndarray, size (n, p)
            input data

        Returns
        -------
        tuple
            list of outputs of each layer, proba of the classification layer
        """
        liste_sorties = [X]
        for rbm in self.DBN.RBM_list:
            sortie_RBM = rbm.entree_sortie(liste_sorties[-1])
            liste_sorties.append(sortie_RBM)

        sortie_DBN = liste_sorties[-1]

        sortie_RBM = self.RBM_classif.entree_sortie(sortie_DBN)
        liste_sorties.append(sortie_RBM)
        proba = self.RBM_classif.calcul_softmax(sortie_DBN)

        return liste_sorties, proba

    def retropropagation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_epochs: int,
        learning_rate: float,
        batch_size: int,
    ):
        """Train the DNN model with backpropagation.

        Parameters
        ----------
        X : np.ndarray, size (n, p)
            input data
        y : np.ndarray, size (n)
            input label
        n_epochs : int
        learning_rate : float
        batch_size : int
        """
        X = X.copy()
        y = y.copy()
        loss_history = []
        acc_history = []
        with tqdm(range(n_epochs)) as pbar:
            for _ in pbar:
                X, y = shuffle(X, y)

                for batch in range(0, X.shape[0], batch_size):
                    X_batch = X[batch : min(batch + batch_size, X.shape[0])]
                    y_batch = y[batch : min(batch + batch_size, X.shape[0])]
                    tb = X_batch.shape[0]

                    liste_sorties, _ = self.entree_sortie_reseau(X_batch)

                    y_onehot = np.eye(liste_sorties[-1].shape[1])[y_batch]
                    dL_dxp_tilde = liste_sorties[-1] - y_onehot
                    cp = dL_dxp_tilde

                    xp_moins_1 = liste_sorties[-2]

                    dL_dWp = xp_moins_1.T @ cp
                    dL_dbp = cp.sum(axis=0)

                    self.RBM_classif.W -= (learning_rate / tb) * dL_dWp
                    self.RBM_classif.b -= (learning_rate / tb) * dL_dbp

                    W_plus_1 = self.RBM_classif.W
                    cp_plus_1 = cp
                    xp = xp_moins_1
                    for p in range(len(self.DBN.RBM_list) - 1, -1, -1):
                        xp_moins_1 = liste_sorties[p]
                        cp = (cp_plus_1 @ W_plus_1.T) * (xp * (1 - xp))

                        dL_dWp = xp_moins_1.T @ cp
                        dL_dbp = cp.sum(axis=0)

                        self.DBN.RBM_list[p].W -= (learning_rate / tb) * dL_dWp
                        self.DBN.RBM_list[p].b -= (learning_rate / tb) * dL_dbp

                        W_plus_1 = self.DBN.RBM_list[p].W
                        cp_plus_1 = cp
                        xp = xp_moins_1

                liste_sorties, proba = self.entree_sortie_reseau(X)
                loss = cross_entropy(proba, y)
                pred = proba.argmax(axis=1)
                acc = (pred == y).sum() / len(pred)
                loss_history.append(loss)
                acc_history.append(acc)
                pbar.set_description(f"loss {loss:.4f} - acc {acc:.3f} ")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(loss_history, label="Loss")
        axes[1].plot(acc_history, label="Accuracy")
        for ax in axes:
            ax.legend()
            ax.grid()
        plt.show()

    def test(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Test the DNN model on X_test and y_test.

        Parameters
        ----------
        X : np.ndarray, size (n, p)
            test data
        y : np.ndarray, size (n)
            test label

        Returns
        -------
        tuple[float, float]
            loss and accuracy
        """
        _, proba = self.entree_sortie_reseau(X)
        loss = cross_entropy(proba, y)
        pred = proba.argmax(axis=1)
        acc = (pred == y).sum() / len(pred)
        print(f"loss {loss:.4f} - acc {acc:.3f} ")
        return loss, acc


def cross_entropy(proba: np.ndarray, y: np.ndarray) -> float:
    """Cross entropy loss function.

    Parameters
    ----------
    proba : np.ndarray, size (n, n_classes)
    y : np.ndarray, size (n)

    Returns
    -------
    float
    """
    output = 0
    for i in range(proba.shape[0]):
        proba_i = proba[i]
        y_i = y[i]
        output -= np.log(proba_i[y_i])
    return output / proba.shape[0]
