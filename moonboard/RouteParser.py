import json
import urllib.error
from html.parser import HTMLParser
from urllib.request import Request, urlopen

def ParseProblem(link: str) -> str:
    problemDict = {}
    WebSocHTML = Request(link, headers={'User-Agent': 'Mozilla/5.0'})
    with urlopen(WebSocHTML) as file:
        for line in file:
            decodedLine = line.decode()
            if(decodedLine.startswith('    var problem = JSON.parse')):
                decodedLine = decodedLine[30:-5]
                problem = json.loads(decodedLine)
                problemDict['name'] = problem['Name']
                print(problem['Name'])
                problemDict['ID'] = problem['Id']
                problemDict['setterGrade'] = problem['Grade']
                problemDict['userGrade'] = problem['UserGrade']
                problemDict['userRating'] = problem['UserRating']
                problemDict['repeats'] = problem['Repeats']
                problemDict['isBenchmark'] = problem['IsBenchmark']
                problemDict['moves'] = []
                moves = problem['Moves']
                holdsString = '{}\n'.format(len(moves))

                for hold in moves:
                    tempHoldString = '{} {} {}\n'.format(hold['Description'], hold['IsStart'], hold['IsEnd'])
                    problemDict['moves'].append(tempHoldString)


                return problemDict
                #problemString = '{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n'.format(name, ID, setterGrade, userGrade, userRating, repeats, isBenchmark, holdsString)


#	Name
#	ID
#	Setter Grade
#	User Grade
#	User Rating
#	Repeats
#	Is Benchmark
#	Holds Count
#	Location IsStart IsEnd
#	Rest of holds, same format as previous line