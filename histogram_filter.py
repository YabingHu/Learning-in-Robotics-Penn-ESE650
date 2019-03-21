import numpy as np


class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """

    def histogram_filter(self, cmap, belief, action, observation):
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        Use starter.npz data given in /data to debug and test your code. starter.npz has sequence of actions and 
        corresponding observations. True belief is also given to compare your filters results with actual results. 
        cmap = arr_0, actions = arr_1, observations = arr_2, true state = arr_3
    
        ### Your Algorithm goes Below.
        '''
        sensor_right = 0.9
        sensor_wrong = 1 - sensor_right
        p_move = 0.9
        p_stay = 1 - p_move
        m,n=cmap.shape[1],cmap.shape[0]

        def actionmodel(new_bel, action):
            bel = np.zeros([m,n])
            for i in range(m):
                    for j in range(n):
                        if action[1]==0:
                            if  j+action[0]<0 or j+action[0]>cmap.shape[1]-1:
                                bel[i][j]+=new_bel[i][j] 
                            else:
                                bel[i][j]+=p_stay*new_bel[i][j]
                                bel[i][j+action[0]]+= p_move*new_bel[i][j]
                                
                        if action[0]==0:
                            if i-action[1]<0 or i-action[1]>cmap.shape[0]-1:
                                bel[i][j]+=1*new_bel[i][j]
                            else:
                                bel[i][j]+=p_stay*new_bel[i][j]
                                bel[i-action[1]][j]+= p_move*new_bel[i][j]      
            return bel


        def sense(new_bel, Z):
            bel = np.zeros([m,n])
            eta = 0
            for i in range(m):
                for j in range(n):
                    hit =int(cmap[i][j] == Z)
                    bel[i][j] = new_bel[i][j]*(hit*sensor_right + (1 - hit)*sensor_wrong)
                    eta += bel[i][j]
            bel = bel/eta 
            return bel

        bel=actionmodel(belief,action) 
        bel=sense(bel, observation)
        idx = np.argwhere(bel.max() == bel)[0]
        idx[0], idx[1] = idx[1], bel.shape[0] - 1-idx[0] 
        return bel,idx
        
