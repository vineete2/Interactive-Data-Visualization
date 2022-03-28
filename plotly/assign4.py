
import pandas as pd
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.express as px



from pandas.plotting import parallel_coordinates

#Read data from csv with required fields

df = pd.read_csv("DataWeierstrass.csv", skiprows =1 , delimiter = ';',names=['professor', 'lecture', 'participants', 'professional expertise',
       'motivation', 'clear presentation', 'overall impression'])

# Average the scores for each professor as multiple ratings exists

data=df.groupby('professor')[['participants', 'professional expertise','motivation', 'clear presentation', 'overall impression']].mean()

# Add column Professor into data as column
data1=data
data1.index.name = 'professor'
data1.reset_index(inplace=True)

#Filter the data as we do not require the grades above 2 and simplifies visualisation
# As per grading system: 1 is the best and 6 is the worst grade the professors, the values greater than 2 can be ignored
filterData = data1[(data1['professional expertise'] < 2) & (data1['motivation'] < 2) & (data1['clear presentation'] < 2) & (data1['overall impression'] < 2)]

#Task1 Visualize given data with a scatterplot matrix

fig = px.scatter_matrix(filterData,
                        dimensions=['participants', 'professional expertise','motivation', 'clear presentation', 'overall impression']
                        , color=filterData['professor']
                        , width=800,
                        height=800
                        , symbol=filterData['professor']
                        )
#fig.show()
#fig.show()


fig.update_layout(
    title={
        'text': "DataWeierstrass award",
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

#fig.write_image("fig1.png")
fname="scatter_matrix"
pio.write_image(fig,fname+'.png')

#Task2 Visualize given data with parallel coordinates.
#print(filterData['professor'])

parallel_coordinates(filterData, class_column='professor', cols=['professional expertise', 'motivation', 'clear presentation', 'overall impression'])

plt.title('DataWeierstrass award')
plt.ylabel('Scores 1:best, 2:good, 6:worst');
plt.legend(bbox_to_anchor=(1, 1.05))

plt.savefig('parallel.png', dpi=300, bbox_inches='tight')


