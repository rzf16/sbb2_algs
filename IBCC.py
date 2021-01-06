import numpy as np
from scipy.special import psi


class IBCC:
    # Class for performing Independent Bayesian Classifier Combination

    def __init__(self, confusion_prior_init, class_prob_prior_init,
                 convergence_check_freq=1, convergence_threshold=0.0001, max_iterations=500):
        """
        Initializes the IBCC combiner given initial priors

        @param dict confusion_prior_init: initial Dirichlet priors for the confusion matrix of each classifier
        @param list class_prob_prior_init: initial Dirichlet prior for class probabilities
        """
        self.confusion_prior_init = confusion_prior_init
        for classifier in self.confusion_prior_init:
            self.confusion_prior_init[classifier] = np.array(self.confusion_prior_init[classifier]).astype(float)
        self.class_prob_prior_init = np.array(class_prob_prior_init).astype(float)
        self.num_classes = len(class_prob_prior_init)

        self.ev_init = self.class_prob_prior_init / np.sum(self.class_prob_prior_init, axis=0)
        self.ln_confusion_init = {}
        for classifier, confusion_prior in self.confusion_prior_init.items():
            self.ln_confusion_init[classifier] = np.zeros(confusion_prior.shape)
            psi_sum_confusion_prior = psi(np.sum(confusion_prior, axis=1))
            for j in range(self.num_classes):
                self.ln_confusion_init[classifier][:,j] = psi(confusion_prior[:,j]) - psi_sum_confusion_prior
        self.ln_class_prob_init = psi(self.class_prob_prior_init) - psi(np.sum(self.class_prob_prior_init))

        self.convergence_check_freq = convergence_check_freq
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations

    def inferVB(self, anomaly_scores_in):
        """
        Runs Variational Bayes inference using given priors and anomaly scores

        @param dict anomaly_scores_in: probability of normal and anomaly classes given by each classifier for a frame
        @return float ev: eexpected value estimate for each class
        """
        # Initialize parameters
        confusion_prior = self.confusion_prior_init
        class_prob_prior = self.class_prob_prior_init
        ev = self.ev_init
        ln_confusion = self.ln_confusion_init
        ln_class_prob = self.ln_class_prob_init

        anomaly_scores = anomaly_scores_in
        for classifier in anomaly_scores_in:
            anomaly_scores[classifier] = np.array(anomaly_scores_in[classifier]).astype(float)

        iterations = 0
        while iterations < self.max_iterations:
            prev_ev = ev
            ev = self._updateEV(ln_confusion, ln_class_prob, anomaly_scores)
            ln_class_prob, class_prob_prior = self._updateLnClassProb(ev)
            ln_confusion, confusion_prior = self._updateLnConfusion(anomaly_scores, ev)

            # Check convergence
            if iterations % self.convergence_check_freq == 0:
                ev_delta = max(abs(ev - prev_ev))
                # print("Change: " + str(ev_delta))
                if ev_delta < self.convergence_threshold:
                    break
            
            iterations += 1
        
        # print("IBCC converged after " + str(iterations) + " iterations")
        return ev


    def _updateLnClassProb(self, ev):
        """
        Updates the log class probabilities

        @param ndarray ev: current expected value estimate for each class
        @return updated_ln_class_prob: updated log class probabilities estimate
        @return updated_class_prob_prior: updated Dirichlet prior for class probabilities
        """
        # Update class probability priors
        updated_class_prob_prior = self.class_prob_prior_init
        for i in range(self.num_classes):
            updated_class_prob_prior[i] += ev[i]

        # Update log class probabilities
        updated_ln_class_prob = psi(updated_class_prob_prior) - psi(np.sum(updated_class_prob_prior))

        return updated_ln_class_prob, updated_class_prob_prior

    def _updateLnConfusion(self, anomaly_scores, ev):
        """
        Updates the log confusion matrix estimate for each classifier

        @param dict anomaly_scores: probability of normal and anomaly classes given by each classifier for a frame
        @param ndarray ev: current expected value estimate for each class
        @return dict updated_ln_confusion: updated log confusion matrix estimate for each classifier
        @return dict updated_confusion_prior: updated Dirichlet priors for the confusion matrix of each classifier
        """
        # Update confusion priors
        updated_confusion_prior = self.confusion_prior_init
        for classifier in updated_confusion_prior:
            for i in range(self.num_classes):
                updated_confusion_prior[classifier][i,:] += anomaly_scores[classifier] * ev[i]

        # Update log confusion
        updated_ln_confusion = {}
        for classifier, confusion_prior in updated_confusion_prior.items():
            updated_ln_confusion[classifier] = np.zeros(confusion_prior.shape)
            psi_sum_confusion_prior = psi(np.sum(confusion_prior, axis=1))
            for j in range(self.num_classes):
                updated_ln_confusion[classifier][:,j] = psi(confusion_prior[:,j]) - psi_sum_confusion_prior

        return updated_ln_confusion, updated_confusion_prior


    def _updateEV(self, ln_confusion, ln_class_prob, anomaly_scores):
        """
        Updates the expected value estimate of each class using current confusion and class probabilities

        @param dict ln_confusion: current log confusion matrix estimate for each classifier
        @param ndarray ln_class_prob: current log class probabilities estimate
        @param dict anomaly_scores: probability of normal and anomaly classes given by each classifier for a frame
        @return ndarray ev: updated expected value estimate for each class
        """
        ln_joint = self._calcLnJoint(ln_confusion, ln_class_prob, anomaly_scores)
        joint = np.exp(ln_joint)
        ev = joint / np.sum(joint)
        return ev

    def _calcLnJoint(self, ln_confusion, ln_class_prob, anomaly_scores):
        """
        Computes log joint likelihood of the current parameter estimates given observations

        @param dict ln_confusion: current log confusion matrix estimate for each classifier
        @param ndarray ln_class_prob: current log class probabilities estimate
        @param dict anomaly_scores: probability of normal and anomaly classes given by each classifier for a frame
        @return ndarray ln_joint: log joint likelihood of the current parameter estimates for each class given observations
        """
        ln_joint = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            ln_joint[i] = ln_class_prob[i]
            for classifier, confusion in ln_confusion.items():
                ln_joint[i] += np.sum(np.multiply(anomaly_scores[classifier], confusion[i,:]))
        return ln_joint