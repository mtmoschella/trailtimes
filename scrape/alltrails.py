from tqdm import tqdm
import hashlib
import urllib.request
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import re
import pandas as pd
import os
import time

metadata_columns = ['Title', 'Distance (mi)', 'Elevation Gain (ft)', 'Route Type', 'Number of Recordings']
data_columns = ['Athlete ID', 'Start', 'Stop', 'Total Time (s)', 'Moving Time (s)', 'Distance (mi)', 'Elevation Gain (ft)', 'Elevation Loss (ft)']
parse_columns = ['firstName', 'lastName', 'dateTimeStart', 'dateTimeStop', 'timeTotal', 'timeMoving', 'distanceTotal', 'elevationGain', 'elevationLoss']
meters_to_miles = 0.0006213711922373339
meters_to_feet = 3.280839895013123

# constants for parsing activity url
root_url = 'https://www.alltrails.com/explore/recording/'
activity_start_flag = 'href="/explore/recording/'
activity_stop_flag = '"'

def getwebdriver(url):
    options = Options()
    options.add_argument('--headless')
    driver = webdriver.Firefox(options=options)
    driver.get(url)
    driver.implicitly_wait(5)
    return driver

def trim(string):
    """
    Removes arbitrary number of occurences of the string "&quot;" from beginning and end of string

    Returns the trimmed string
    """
    bad = "&quot;"
    # remove from beginning
    while string[:len(bad)] == bad:
        string = string[len(bad):]

    # remove from end
    while string[-len(bad):] == bad:
        string = string[:-len(bad)]

    return string

def remove_substring(string, substring, replace=''):
    """
    Removes commas from string
    """
    while substring in string:
        i = string.index(substring)
        string = string[:i]+replace+string[i+len(substring):]
    return string

def get_full_html(driver, max_clicks):
    """
    Given driver at an alltrails.com webpage, click "Recordings" and then click "Show more recordings" until the button disappears or until max_clicks is reached.

    Returns the resulting rendered html as a string.
    """
    # click Recordings
    button = driver.find_element_by_xpath('//button[contains(.,"Recordings")]')
    text = button.text
    button.click()
    
    # click Show more recordings until the button disappears
    for i in tqdm(range(max_clicks), ascii=True, desc="Loading Recordings"):
        try:
            # click button if button exists
            button = driver.find_element_by_xpath('//button[contains(@title,"Show more recordings")]')
            button.click()
            time.sleep(5) # wait before next click to ensure everything loads properly
        except:
            # if button doesn't exist, stop
            break

    html = driver.execute_script("return document.getElementsByTagName('html')[0].innerHTML")
    return html

def get_activities(html):
    """
    Given the full website html as a string, return the unique activity strings that are found.

    Activity string is defined as href="/explore/recording/<activity>"

    Returns a list of unique acitivity strings
    """
    # loop over available activity data
    activities = []
    pbar = tqdm(total=html.count(activity_start_flag), ascii=True, desc="Identifying Activities")
    i=0
    while activity_start_flag in html[i+1:]:
        i = html.index(activity_start_flag, i+1)
        j = i + len(activity_start_flag)-1
        k = html.index('"', j)
        activities.append(html[j+1:k])
        pbar.update(1)
    pbar.close()

    return list(set(activities)) # return only unique activities

def parse_activity(activity):
    """
    Given the alltrails.com/ url-name for the activity, get the url,
    and scrape the data specified in <parse_columns>

    Everything follows the pattern of *<column_name>*:*<value>*,
    where * denotes arbitrary amounts of occurrences of the string "&quot;"

    Returns a dict with the parsed data (columns given by data_columns)
    """

    # init
    parsed_data = dict()

    ntries = 0
    maxtries = 10
    while ntries<maxtries:
        try:
            ntries +=1
            # fetch html as a string
            url = root_url + activity
            parser = urllib.request.urlopen(url)
            html = ''
            for line_utf in parser.readlines():
                html += line_utf.decode('utf8')
            parser.close()
            break
        except:
            print("WARNING: Failed to load HTML.")
            print("Trying to fetch HTML again ("+str(ntries)+" of "+str(maxtries)+")")
    
    # loop over columns to parse
    for col in parse_columns:
        try:
            i = html.index(col) 
            j = html.index(':', i)
            k = html.index(',', j)
            parsed_data[col] = trim(html[j+1:k].strip('} '))
        except:
            #print("WARNING: Found missing column "+col)
            parsed_data[col] = ''

    # convert from parsed data columns to regular data columns
    data = dict()
    data['Athlete ID'] = str(hashlib.sha512((parsed_data['firstName']+' '+parsed_data['lastName']).encode('utf8')).hexdigest()) # hash name for security, string
    data['Start'] = parsed_data['dateTimeStart'] # date-time string
    data['Stop'] = parsed_data['dateTimeStop'] # date-time string
    data['Total Time (s)'] = float(parsed_data['timeTotal']) # seconds, float
    data['Moving Time (s)'] = float(parsed_data['timeMoving']) # seconds, float
    data['Distance (mi)'] = float(parsed_data['distanceTotal'])*meters_to_miles # miles, float
    data['Elevation Gain (ft)'] = float(parsed_data['elevationGain'])*meters_to_feet # ft, float
    data['Elevation Loss (ft)'] = float(parsed_data['elevationLoss'])*meters_to_feet # ft, float
    return data

def parse_metadata(driver):
    """
    Parses the metadata from the already loaded webpage
    
    Returns the metadata dict (columns given by metadata_columns)
    """
    metadata = dict()
    html = driver.page_source
    
    # get title
    i = html.index('<title')
    j = html.index('>', i)
    k = html.index('</title>')
    title = html[j+1:k].strip(' ')
    if '\n' in title:
        title = title[:title.index('\n')]
    metadata['Title'] = [title]
    
    # get distance, elevation, route type
    trailstat_flag = '<span class="styles-module__trailStatIcons___3yGjQ">'
    sections = html.split(trailstat_flag)
    assert len(sections)==4, "ERROR: unexpected number of occurences for trailstat_flag"

    # get length of the route
    distance_string = sections[1]
    i = distance_string.index('</span>')
    j = distance_string.index('>', i+7)
    k = distance_string.index('</span>',j)
    distance, distance_unit = distance_string[j+1:k].strip(' ').split(' ')
    metadata['Distance (mi)'] = [float(remove_substring(distance, ','))]
    if distance_unit!='mi':
        print("WARNING: unexpected unit for distance ("+distance_unit+")")

    # get elevation gain
    elevation_string = sections[2]
    i = elevation_string.index('</span>')
    j = elevation_string.index('>', i+7)
    k = elevation_string.index('</span>',j)
    elevation, elevation_unit = elevation_string[j+1:k].strip(' ').split(' ')
    metadata['Elevation Gain (ft)'] = [float(remove_substring(elevation, ','))]
    if elevation_unit !='ft':
        print("WARNING: unexpeted unit for elevation ("+elevation_unit+")")

    # get route type
    route_string = sections[3]
    i = route_string.index('</span>')
    j = route_string.index('>', i+7)
    k = route_string.index('</span>',j)
    route = remove_substring(route_string[j+1:k].strip(' '), 'amp;')
    metadata['Route Type'] = [route]
    
    # get the number of recordings
    i = html.index('Recordings (')
    j = html.index('(', i)
    k = html.index(')',j)
    nrecordings = html[j+1:k].strip(' ')
    nrecordings = int(remove_substring(nrecordings,','))
    metadata['Number of Recordings'] = [nrecordings]

    return metadata

def parse_route(url):
    """
    Given the url for an alltrails.com route, parse its title and metadata,
    as well as data from all available recordings.

    Everything follows the pattern 'href="/explore/recording/<activity_name>"'

    Returns a metadata dict (columns given by metadata_columns), 
    as well as a data dict (columns given by data_columns)
    """

    good = False
    ntries = 0
    maxtries = 10
    while not good and ntries<maxtries:
        ntries += 1
        
        driver = getwebdriver(url)

        # get metadata
        metadata = parse_metadata(driver)
        nrec = metadata['Number of Recordings'][0]

        # get html
        max_clicks = 1 + int(nrec/30)
        html = get_full_html(driver, max_clicks)
        driver.close()
        
        # get list of activities
        activities = get_activities(html)

        if len(activities)==nrec:
            good = True # termination condition
        else:
            print("WARNING: Found "+str(len(activities))+" activities but expected "+str(nrec))
            print("Trying to fetch HTML again ("+str(ntries)+" of "+str(maxtries)+")")

    if not good:
        raise Exception("ERROR: could not fetch HTML with correct number of activities")
    else:
        assert len(activities)==nrec, 'ERROR: in my logic somewhere'

    data = dict([ (col, []) for col in data_columns])
    for activity in tqdm(activities, ascii=True, desc="Parsing Activities"):
        activity_data = parse_activity(activity)
        for col in data_columns:
            data[col].append(activity_data[col])

    return metadata, data

def scrape(url, overwrite=False):
    shortname = url[url.rfind('/')+1:]
    datadir = '../data/'+shortname+'/'
    metafile = datadir+shortname+'_metadata.csv'
    datafile = datadir+shortname+'.csv'
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    if not os.path.exists(metafile) or not os.path.exists(datafile) or overwrite:
        print("Scraping AllTrails Route: "+shortname)
        metadata, data = parse_route(url)
        pd.DataFrame(data).to_csv(datafile, index=False)
        pd.DataFrame(metadata).to_csv(metafile, index=False) # index=[0] because metadata contains scalars (i.e. its entries are not arrays)
        print("Done.")
    else:
        print("Specified data already exists. Do you want to set overwrite=True?")
    return

def crawl(url):
    """
    Crawls a page containing many routes, identifies routes, and then scrapes each route page
    """
    # implement me
    pass

if __name__=='__main__':
    #route_url = "https://www.alltrails.com/trail/us/colorado/green-mountain-west-trail"
    #route_url = "https://www.alltrails.com/trail/us/utah/angels-landing-trail"
    #route_url = "https://www.alltrails.com/trail/us/new-jersey/croft-farm-trail--6"
    #route_url = "https://www.alltrails.com/trail/us/california/half-dome-trail"
    #route_url = "https://www.alltrails.com/trail/us/virginia/old-rag-mountain-loop-trail"
    #route_url = "https://www.alltrails.com/trail/us/wyoming/cascade-canyon-trail"
    #route_url = "https://www.alltrails.com/trail/us/colorado/emerald-lake-trail"
    #route_url = "https://www.alltrails.com/trail/us/arizona/devils-bridge-trail"
    #route_url = "https://www.alltrails.com/trail/us/tennessee/alum-cave-trail-to-mount-leconte"
    #route_url = "https://www.alltrails.com/trail/us/new-hampshire/mount-lafayette-and-franconia-ridge-trail-loop"
    #scrape(route_url)
    url = "https://www.alltrails.com/us/"
    crawl(url)

