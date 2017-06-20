# Flight-delay-analysis
I studied flight ontime performance data which is accessible at the [Bureau of Transportation Statistics](https://transtats.bts.gov/DatabaseInfo.asp?DB_ID=120&DB_Short_Name=On-Time&DB_Name=Airline%20On-Time%20Performance%20Data&Link=0&DB_URL=Mode_ID=1&Mode_Desc=Aviation&Subject_ID2=0).
Based on all 5.5 million domestic flights from 2016 I generated a  flight delay
prediction model for flights within the USA. The size of the analyzed dataset is 600 MB.
     
An application runs on heroku with a web interface [ONTIMEPREDICTOR](http://flightontimeprediction.herokuapp.com/)
and the code can be found on github [thedatabeat](https://github.com/thedatabeat/Flighdelay).
       
The dataset has 83% ontime flights, therefore if one predicts every flight ontime the
percentage of correctly  predicted ontime flights is 83%.
      
In the model provided, the percentage of correctly predicted ontime flights is 89%.
The area under curve (AUC) score  is 0.67.
       
After testing various machine learning algorithms such as logistic regression,
support vector machines, random forest, neural networks, AdaBoost and gradient boosting,
I choose logistic regression to generate the model.  The other more sophisticated models did not generate better results due to
insufficient structure in the data therefore I applied Occam's razor.
All the analysis is made on python using various libraries such as [pandas](http://pandas.pydata.org/) and
[sklearn](http://scikit-learn.org/).
