class Bias(object):
    """
        This class represents a bias, which is essentially a Beta distribution. 
        Note that this class does not in any way compute sentiments or topics for tweets.
        Instead, an instance Bias class should be used for each (user, topic) pairs in 
        the system.
    """
    def __init__(self, alpha0=1, beta0=1):
        """
            Creates the new bias distribution.
            :param alpha: The alpha parameter of the bias prior (default: 1). Can be set to previously inferred value to update the bias.
            :param beta: The beta parameter of the bias prior (default: 1). Can be set to previously inferred value to update the bias.
            :return: alpha as float, beta as float
        """
        # Initial beta distribution parameters
        self.alpha = alpha0
        self.beta = beta0

    def get_bias_distribuion(self):
        """
            Get the parameters of the bias distribution (i.e.: alpha and beta parameters).
            :return: alpha as float, beta as float
        """
        return self.alpha, self.beta

    def get_bias(self):
        """
            Get the MAP estimate of the bias.
            :return: bias as float
        """
        return self.alpha/(self.alpha + self.beta)

    def naive_infer(self, sentiments: list):
        """
            Compute the posterior distribution of the bias by updating the distribution's parameters according to the evidence.
            :param sentiments: list of sentiments (int) with 0 indicating a negative sentiment and 1 indicating a positive sentiment. 
        """
        # Check if all sentiments are 0 or 1
        if any([s != 0 and s != 1 for s in sentiments]):
            raise ValueError("Sentiments contain invalid values (all sentiments must be 0 or 1)")

        s = sum(sentiments)
        self.alpha = self.alpha + s
        self.beta = self.beta + len(sentiments) - s

    def em_infer(self, sentiments: list):
        """
            Compute the posterior distribution of the bias by updating the distribution's parameters according to EM-like weighted evidence.
            :param sentiments: list of sentiments predictions (float) with 0.0 indicating a certain negative sentiment and 1.0 indicating a certain positive sentiment.
        """
        pos_sentiments = [s * 1.0 for s in sentiments]
        neg_sentiments = [s * 0.0 for s in sentiments]

        self.alpha = self.alpha + sum(pos_sentiments)
        self.beta = self.beta + sum(neg_sentiments)

    def infer2(self, sentiments: list, specificity: float = 1.0, sensitivity: float = 1.0):
        pos_sentiments = [specificity if s == 1 else (1-sensitivity) for s in sentiments]
        neg_sentiments = [(1-specificity) if s == 1 else sensitivity for s in sentiments]

        self.alpha = self.alpha + sum(pos_sentiments)
        self.beta = self.beta + sum(neg_sentiments)

def test():
    sent_pred = [0.9, 0.7, 0.8, 0.51, 0.4, 1, 0.1, 0.97,.88,.95,.46]

    bias1 = Bias()
    bias1.naive_infer([round(s) for s in sent_pred])
    print(f"Naive bias: {bias1.get_bias()}")
    print(f"Naive dist: {bias1.get_bias_distribuion()}")

    print('=' * 50)
    bias2 = Bias(alpha0=1, beta0=1)
    bias2.em_infer(sent_pred)
    print(f"EM bias: {bias2.get_bias()}")
    print(f"EM dist: {bias2.get_bias_distribuion()}")

    print('=' * 50)
    bias3 = Bias(alpha0=1, beta0=1)
    bias3.infer2([round(s) for s in sent_pred], specificity=0.9, sensitivity=0.75)
    print(f"Spec/sens bias: {bias3.get_bias()}")
    print(f"Spec/sens dist: {bias3.get_bias_distribuion()}")


if __name__ == "__main__":
    test()