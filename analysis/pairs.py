import pandas as pd
import os
from tqdm import tqdm

datadir = '../data/'

@dataclass
class Actvity:
    """
    Represents a single actvity done by an athlete
    """
    route : str 
    athleteid : str
    start : str # date-time string
    stop : str # date-time string
    totaltime : float # seconds
    movingtime : float # seconds
    distance : float # miles
    gain : float # feet
    loss : float # feet

class ActivityList:
    """
    Represents a mutable list of multiple activities
    """

    data = {'route', 'athleteid', 'start', 'stop', 'totaltime', 'movingtime', 'distance', 'gain', 'loss'}
    
    def __init__(self, activities=[]):
        self.routes = []
        self.athleteids = []
        self.starts = []
        self.stops = []
        self.totaltime = []
        self.
class Pair:
    """
    Represents an ordered pair of activities performed by the same athlete on different routes
    """

    def __init__(self, ax, ay):
        """
        ax, ay: instances of class Activity
        """
        if ax.athleteid != ay.athleteid:
            raise Exception("ERROR: ax and ay must belong to the same athlete")
        self.athleteid = ax.athleteid
        self.ax = ax
        self.ay = ay
        
    def getRoutes(self):
        """
        Returns the two uniue routes in this pair (ordered)
        """
        return self.ax.route, self.ay.route
    
    def getTimes(self):
        """
        Returns the ordered pair of times for the routes in the pair
        """
        return self.ax.totaltime, self.ay.totaltime

    def getAthleteID(self):
        """
        Returns the ID of the athlete that this pair belongs to
        """
        return self.athleteid
    
class Athlete:
    """
    Represents a paired athlete, with at least 2 activities on 2 different routes.
    
    """
    def __init__(self, activities, athleteid=None):
        """
        activities: a list of instances of class Activity
        """
        # check valid args
        if len(activities)==0:
            raise Exception("ERROR: activities cannot be empty")
        if athleteid is None:
            self.athleteid = activities[0].athleteid
        else:
            self.athleteid = athleteid
        
            
    def getRoutes(self):
        """
        Returns a list of routes the athlete has done
        """
        pass

    def getActivities(self, route=None):
        """
        Returns a list of Activity objects
        """
        pass

    def getPairs(self, routex=None, routey=None)
        """
        Returns a list of Pairs for the specified routes
        or return all pairs if routex==routey==None
        """
        pass

    
class Model:
    """
    Represents a simultaneous regression of all possible route combinations.
    Think of this like a corner plot, if there are N routes, there are N*(N-1)/2
    unique comparison that can be made.
    """
    def __init__(self):
        pass

    def residuals(self):
        """
        Compute the sum of the squares of all regression residuals
        """

def get_pairs():
    """
    Get activity pairs (same athlete, different routes) across all available data.

    Returns a dictionary indexed by the "Athlete ID" and containing a set of route names
    """
    # build dict of athletes with a set of routes by each athlete
    athletes = dict()
    routes = os.listdir(datadir)
    for route in routes:
        fname = datadir+route+'/'+route+'.csv'
        data = pd.read_csv(fname)
        ids = data['Athlete ID'].unique()
        for i in ids:
            if i in athletes.keys():
                athletes[i].add(route)
            else:
                athletes[i] = set([route])

    # remove athletes that don't have multiple routes
    pairs = dict()
    for athlete in athletes.keys():
        if len(athletes[athlete])>1:
            pairs[athlete] = athletes[athlete]
    print("Found "+str(len(pairs))+" out of "+str(len(athletes))+" athletes with multiple routes.")
    return pairs

def get_rankings():
    """
    Builds an ordered ranking of all athletes that have paired activities
    """

    pairs = get_pairs()
    rankings = []
    for athlete in pairs.keys():
        
if __name__=='__main__':
    get_pairs()
