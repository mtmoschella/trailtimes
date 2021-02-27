import pandas as pd
import os
from tqdm import tqdm
from dataclasses import dataclass
import numpy as np

datadir = '../data/'

class Route:
    """
    Represents the metadata of a route
    """

    def __init__(self, routename):
        self.name = routename
        fname = datadir + routename + '/'+routename+'_metadata.csv'
        metadata = pd.read_csv(fname)
        index, row = next(metadata.iterrows()) # get the first row
        self.title = row['Title']
        self.distance = row['Distance (mi)']
        self.gain = row['Elevation Gain (ft)']
        self.routetype = row['Route Type']
        self.nrecordings = row['Number of Recordings']
        return

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return str(self.name)

@dataclass(frozen=True)
class Activity:
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

    def from_row(route, row):
        """
        Given a pandas dataframe row (and the route), return the activity it corresponds to
        """
        return Activity(route=route,
                        athleteid=row['Athlete ID'], start=row['Start'], stop=row['Stop'],
                        totaltime=row['Total Time (s)'], movingtime=row['Moving Time (s)'],
                        distance=row['Distance (mi)'],
                        gain=row['Elevation Gain (ft)'], loss=row['Elevation Loss (ft)'])

    def getMetadata(self):
        return Route(self.route)
        
    def clean(self):
        """
        Returns True if the activity passes cuts for its route
        """

        meta = self.getMetadata()
        # currently, require distance to agree within 10% and totaltime>0
        return np.absolute((self.distance-meta.distance)/meta.distance)<0.1 and self.totaltime>0.

class Pair:
    """
    Represents a (non-ordered) pair of activities performed by the same athlete on different routes
    """

    def __init__(self, ax, ay):
        """
        ax, ay: instance of class Activity

        ax and ay must have the same athleteid
        ax and ay must not correspond to the same route
        """
        if ax.athleteid != ay.athleteid:
            raise Exception("ERROR: ax and ay must belong to the same athlete")
        if ax.route == ay.route:
            raise Exception("ERROR: ax and ay must represent different routes")
        
        self.athleteid = ax.athleteid
        self.ax = ax
        self.ay = ay

    def getRoutes(self):
        """
        Returns the two uniue routes in this pair (ordered)
        """
        return frozenset([self.ax.route, self.ay.route]) # return hashable

    def getTimes(self):
        """
        Returns a dict of pair of times indexed by routes for the activities in the pair
        """
        return {self.ax.route : self.ax.totaltime, self.ay.route : self.ay.totaltime}

    def getTime(self, route):
        if route==self.ax.route:
            return self.ax.totaltime
        elif route==self.ay.route:
            return self.ay.totaltime
        else:
            raise Exception("ERROR: pair does not contain the route "+str(route))

    def getCoord(self, routex, routey):
        """
        Returns the times as an (x,y) ordered pair (tuple)
        """
        if routex==self.ax.route and routey==self.ay.route:
            return self.ax.totaltime, self.ay.totaltime
        elif routex==self.ay.route and routey==self.ax.route:
            return self.ay.totaltime, self.ax.totaltime
        else:
            raise Exception("ERROR: invalid route specification")
        
    def getAthleteID(self):
        """
        Returns the ID of the athlete that this pair belongs to
        """
        return self.athleteid

class Pairs:
    """
    Represents a collection of pairs.
    """

    def __init__(self, pairs=[]):
        """
        pairs: a list of pairs
        """
        self.pairs = set(pairs) # for now, just implement as a set of pairs
        return

    def add(self, pair):
        """
        Adds the pair to the collection
        """
        self.pairs.add(pair)

    def remove(self, pair):
        """
        Removes the pair from the collection, raises error if pair is not in the collection.
        """
        self.pairs.remove(pair)

    def getRoutes(self):
        """
        Returns a set of sets of pair routes, i.e. {{routex, routey}, ... }
        """
        output = set()
        for pair in self.pairs:
            output.add(pair.getRoutes())
        return output
    
    def getPairs(self, routex=None, routey=None, either=False):
        """
        Returns a set of pairs with the specification provided.

        If neither routex nor routey are specified, returns a copy of the entire set of pairs
        If only routex is specified, all returned pairs will have one route == routex.
        If only routey is specified, all returned pairs will have one route == routey.
        If routex and routey are specified, the behavior depends on the either flag.
        If either is True:
              Returns pairs with (routex or routey) in the set of routes
        If either is False (default):
              Returns only pairs with (routex and routey) in the set of routes.
        """
        if routex is None and routey is None:
            return self.pairs.copy()
        else:
            match_routes = set()
            if routex is not None:
                match_routes.add(routex)
            if routey is not None:
                match_routes.add(routey)

            output = set()                
            for pair in self.pairs:
                routes = pair.getRoutes()
                if not either:
                    # entire subset must match
                    if match_routes <= routes:
                        output.add(pair)
                else:
                    # only require one element of subset to match
                    if len(match_routes & routes)>0:
                        output.add(pair)
            return output

class Athlete:
    """
    Represents a paired athlete, with at least 2 activities on 2 different routes.
    
    """
    def __init__(self, activities):
        """
        activities: a list of instances of class Activity
        """
        # check valid args
        if len(activities)==0:
            raise Exception("ERROR: activities cannot be empty")
        self.athleteid = next(iter(activities)).athleteid
        if np.any([ activity.athleteid!=self.athleteid for activity in activities]):
            raise Exception("ERROR: all activities must belong to the same athlete")
        self.activities = set(activities)
            
    def getRoutes(self):
        """
        Returns a set of routes the athlete has done
        """
        routes = {}
        for activity in self.activities:
            routes.add(activity.route)
        return routes

    def getActivities(self, route=None):
        """
        Returns a list of Activity objects
        """
        return self.activities

    def getPairs(self, routex=None, routey=None):
        """
        Returns a set of Pairs for the specified routes
        or return all pairs if routex==routey==None
        """

        if routex is not None and routex==routey:
            raise Exception("ERROR: routex and routey cannot be the same")
            
        pairs = set()
        # this double for loop could be more efficient
        # currently, it double counts all pairs
        for ai in self.activities:
            for aj in self.activities:
                if ai!=aj and ai.route!=aj.route:
                    matches_routex = True if routex is None else ai.route==routex or aj.route==routex
                    matches_routey = True if routey is None else ai.route==routey or aj.route==routey
                    if matches_routey and matches_routex:
                        pairs.add(Pair(ai,aj))
        return pairs

class PairedData:
    """
    Represents a set of data points for a single paired of routes {routex, routey}
    """
    def __init__(self, routex, routey, xdata, ydata):
        if routex==routey:
            raise Exception("ERROR: routex and routey cannot be the same")
        if len(xdata)!=len(ydata):
            raise Exception("ERROR: xdata and ydata must be the same length")
        
        self.routex = routex
        self.routey = routey
        self.xdata = np.array(xdata)
        self.ydata = np.array(ydata)

    def getData(self, route):
        if route==routex:
            return self.xdata
        elif route==routey:
            return self.ydata
        else:
            raise Exception("ERROR: Invalid route "+str(route))
    
class LinearMap2D:
    """
    A functor class representing a mapping between two routes <routex> and <routey>
    """
    def __init__(self, routex, routey, params):
        """
        routex, routey : route strings
        params: a tuple of floats (p0, p1)
        """
        if routex==routey:
            raise Exception("ERROR: routex and routey cannot be the same")
        
        self.routex = routex
        self.routey = routey
        self.params = params

    def f(x, p0, p1):
        return p0 + p1*x

    def finv(y, p0, p1):
        return (y-p0)/p1

    def lstsq(data):
        """
        data: PairedData object

        Returns the least squares parameter solution for the given data as a LinearMap2D object
        """
        x = data.xdata
        y = data.ydata
        matrix = np.transpose([x, np.ones(len(x))])
        params, resid, rank, s = lstsq_matrix(matrix, y, rcond=None)
        return LinearMap2D(data.routex, data.routey, params)
    
    def __eval__(route_orig, route_dest, x_orig):
        if route_orig==routex and route_dest==routey:
            return f(x_orig, *self.params)
        elif route_orig==routey and route_dest==routex:
            return finv(x_orig, *self.params)
        else:
            raise Exception("ERROR: invalid routes specified")

    def compute_residual(self, data):
        """
        data: a PairedData object

        Returns the sum of the squares of the residuals y - f(x)
        """
        xdata = data.getData(routex)
        ydata = data.getData(routey)
        return np.sum((y - f(xdata, *self.params))**2)
    
class LinearModel:
    """
    Represents a simultaneous regression of all possible route combinations.
    Think of this like a corner plot, if there are N routes, there are N*(N-1)/2
    unique comparison that can be made.
    """
    def __init__(self, pairs):
        """
        pairs: a Pairs collection instance
        """
        self.pairs = pairs



    def projection(route_orig, route_dest, maps, xorig):
        """
        route_orig, route_dest: route strings
        maps: a dict of LinearMap2D objects indexed by frozenset([routex, routey])
        xorig: a float
        """
        if route_orig==route_dest:
            return xorig
        else:
            key = frozenset([route_orig, route_dest])
            return maps[key](route_orig, route_dest, xorig)

    def getData(self, routex, routey, project=False, maps=None):
        """
        routex, route: route string
        project: boolean
        maps: a dict of LinearMap2D objects, indexed by frozenset([routex, routey])
        
        Returns a PairedData object
        """
        # check valid args
        if routex==routey:
            raise Exception("ERROR: routex and routey must be different")
        if project and maps is None:
            raise Exception("ERROR: must specify maps if you want to project")
            
        match_routes = {routex, routey}
        xvals = []
        yvals = []
        for pair in self.pairs.getPairs(): # loop over all pairs
            routes = pair.getRoutes()
            vals = dict()
            matched_routes = set(routes & match_routes) 
            if len(matched_routes)==2 or (len(matched_routes)==1 and project):
                # add the matches
                for match in matched_routes:
n                    vals[match] = pair.getTime(match)
                # add the missed matches (need projection)
                if len(matched_routes)==1:
                    route_orig = set(routes-match_routes).pop() # the unpaired route  
                    route_dest = set(match_routes-routes).pop() # the subset of {routex, routey} that is missing
                    vals[route_dest] = projection(route_orig, route_dest, maps, pair.getTime(route_orig))

            ######## remove this assertion statement once I'm convinced the code is safe
            assert set(vals.keys())==match_routes, "ERROR: There is a bug in the code"
            
            xvals.append(vals[routex])
            yvals.append(vals[routey])
            
        return PairedData(routex, routey, xvals, yvals)
    
    def residuals(self, maps, project=False):
        """
        maps: a dict of LinearMap2D objects, indexed by frozenset([routex, routey])

        Compute the sum of the squares of all regression residuals
        """
        s = 0.
        for key in self.pairs.getRoutes():
            routex, routey = key
            data = self.getData(routex, routey, project=project, maps=maps)
            s += maps[key].compute_residual(data)
        return s
    
    def lstsq_solution(self):
        """
        Computes the least squares solution in each 2D plan without projection

        Returns a dict of LinearMap2D objects, indexed by frozenset([routex, routey])
        """
        maps = dict()
        for key in self.pairs.getRoutes():
            routex, routey = key
            data = self.getData(routex, routey)
            maps[key] = LinearMap2D.lstsq(data)
        return maps
    
def get_routes():
    return os.listdir(datadir)

def get_fname(route):
    return datadir+route+'/'+route+'.csv'

def get_activities_by_athlete(clean=True):
    # build dict of athletes with a set of routes by each athlete
    activities = dict()
    routes = get_routes()
    for route in routes:
        fname = get_fname(route)
        data = pd.read_csv(fname)
        for index, row in data.iterrows():
            activity = Activity.from_row(route,row)
            if clean and not activity.clean():
                # don't include unclean activity
                continue
            i = activity.athleteid
            if i in activities.keys():
                activities[i].add(activity)
            else:
                activities[i] = set([activity])
    return activities

def get_athletes():
    """
    Get all athletes

    Returns a dictionary indexed by the "Athlete ID" and containing Athlete objects
    """
    activities = get_activities_by_athlete()
                
    # remove athletes that don't have multiple routes
    athletes = dict()
    for athlete in activities.keys():
        if len(activities[athlete])>1:
            athletes[athlete] = Athlete(activities[athlete]) # sorry this is unreadable
    print("Found "+str(len(athletes))+" out of "+str(len(activities))+" athletes with multiple routes.")
    return athletes

def get_all_pairs():
    """
    Get all pairs in the dataset.

    Returns a Pairs object
    """
    # initialize empty Pairs collection
    pairs = Pairs()

    # fill Pairs
    athletes = get_athletes()
    for athleteid in athletes:
        athlete_pairs = athletes[athleteid].getPairs()
        for pair in athlete_pairs:
            pairs.add(pair)
    return pairs

if __name__=='__main__':
    import matplotlib.pyplot as plt
    print("Loading Pairs...")
    pairs = get_all_pairs()
    print("Done.")
    routes = pairs.getRoutes()
    print(routes)
    exit()
    model = LinearModel(pairs)
    params = model.lstsq_solution()
    for routex in pairs.keys():
        for routey in pairs[routex].keys():
            print(routex, routey)
            continue
            p = params[routex][routey]
            n = len(pairs[routex][routey])
            if n<2:
                continue
            xvals = np.zeros(n)
            yvals = np.zeros(n)
            for i, pair in enumerate(pairs[routex][routey]):
                xvals[i] = pair.getTime(routex)
                yvals[i] = pair.getTime(routey)
            xgrid = np.linspace(np.amin(xvals), np.amax(xvals), 1000)
            ygrid = model.f(xgrid, *p)
            plt.figure()
            plt.scatter(xvals, yvals, color='black', marker='o')
            plt.plot(xgrid, ygrid, color='blue')
            plt.show()
