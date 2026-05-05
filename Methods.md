# Methods

For this project there are a lot of tools and methods to consider, since football has started to become more data driven there are lots of different data to choose from. Also, since machine learning is constantly growing the field is becoming more and more wide and new techniques are constantly forming. 


## Data

### Data types

To provide good models and also a good baseline for evaluation the data is very important in this project. The data can be categorized into two separate categories seen below.

* Physical player data
* Event data
* Transfer data

The physical player data used for this project is not much, but consists of age, height and weight. These are physical attributes of a player that always has the ability of making a transfer succeed or not. Usually seasoned players experience a decrease in their physical abilities as they get older, resulting in less statistical outputs. Further, similar articles have been written around the weight and height of athletes in sport.  This is something that gets taken into account by using these metrics for the models.  

The event data, for this project consist of both individual and team data describing actions on the field. For example Goals, Assists, team entries into the box and so on. These are the performance metrics, and give are what is used to construct the player qualities which are also part of the dataset. 

Since the qualities are a weighted sum of multiple performance metrics, they give a rather good idea into the players skill-set compared to looking at only performance metrics. This makes the results more interpretable while also providing less features for the model to handle which lessen chances of for example overfitting. 

The same type of qualities will not only be used as constructors for the player data, but also for constructing the team data. This will make it easier to describe the way a team play and get an idea of how the team plays. Which in turn, results in being able to draw more conclusions when analyzing the models output. Since the team qualities are not present in the dataset, they will be calculated by using the z-score for the team metrics. Which are calculated by grouping them together with the other teams team metrics for the same season and competition. Then by using a weighted average sum the team qualities can be received.

The transfer data is in this case used for the training part of this project. The transfer data will contain teams, competitions, countries, positions that they played for before and after the transfer. This together with the event data provides a good baseline of the model creation.


### Competitions
Competitions, something that makes a big difference during the transfer window in football, and generally for the coming succession of the player when moving teams. All competitions in football have very different styles of play, with some playing quicker football and while others might play more technical. 

To simulate this change, the competitions will be one-hot encoded into the model. However, with a dataset consisting of more than 100.000 rows, and more than 300 competitions it would be insufficient to add all competitions into the model as it would prove to add unnecessary complexity and load to the model.

Therefore the model will only encode competitions that exist in more than 5% of the dataset, due to the fact that if the competition is provided in more than 5% of the dataset there exist enough information to be able to make relative good conclusions on how a transfer to or from that competition usually transfers the players quality.

### Minutes played
Further, looking at the dataset it contains a lot of player transfers, consequently also a lot of players with consistently small amount of minutes. This could mean that players quality values might be a lot improved compared to their real life ability, since the qualities are very often created by using per 90 stats. 

Players playing few minutes while maintaining a slightly better than average output might have very good quality values. But this might be because they are coming off the bench in the dying minutes of the game, when the other players on the field are experiencing a lot of fatigue. Which usually results in the pitch opening up a lot more and players finding more space. 

Furthermore, it might also prove to go the other way with using players playing very few minutes. Teams might have bought younger players, meant to play less minutes and learn the first couple of years in their new team while maybe playing in the second team. This might mean that the player is seen as a unsuccessful transfer when looking at the quality scores, as their output is worse than players that have experienced more minutes. But it might not have been the target teams intent to buy that player to be successful directly.

Therefore, a threshold of 800 minutes played for both the current team and the target team have been set. This provides a player that has played enough to be seen as either successful or unsuccessful in their new team, while keeping enough information to be able to train the models. 

### Missing values
In machine learning the pre-processing of the data is important, and more importantly the ability to handle missing values or outliers is of outmost importance. In this project missing values are something that will be run into often times, as different teams or competitions might have failed to register statistics at a certain point.

When analyzing  the amount of missing values for different columns some columns do have more missing values than others, for example chance prevention and territorial dominance. These qualities are more attributed to defenders, so the likelihood that offensive players have a value in these columns is low since they might not have recorded any statistics in what the quality is built up from. However, there are also some more trivial qualities that are consistent of values that are used over the entirety of the pitch which still contains a lot of missing values, for example run quality.

## Player success model
In this section, a definition and structure of the player success model will be defined. 

### Success
Since this project uses success as its measurement of how good a positional transition is, success is also needed to be determined as it is a subjective context that simply does not have a single meaning. In the case of this project, success is used as two different terms player- and team-success.

First comes player success, which will be a measurement of the players statistics when they move to a team. Since just using raw statistics may be deemed redundant, as it only explains a simple part of the game, the models will instead use player qualities as target variables. Player qualities will be weighted sums of the raw stats, describing a bit more of a player than simply relying on the simple stats. further, the player qualities will be designed and chosen based on the position and the knowledge regarding the position in order to make the models as useful as possible. 

Continuing, table tab:offensive_qualities shows the qualities for the offensive positions striker and winger. These two positions occupies areas that are further up the field and often requires attributes that increase the chance of goals and goal-scoring opportunities. Therefore qualities such as finishing, poaching and progression are part of the target variables. 

| Position | Quality |
| -------- | ------- |
| Striker  | Poaching|
| Striker | FInishing|
| Striker    | Box threat|
| Striker| Effectiveness|
| Winger| Finishing|
| Winger| Progression|
| Winger| Effectiveness|
| Winger| Dribbling|


As well as this, the qualities used as target variables for the midfielders are slightly different compared to the offensive qualities, which can be seen in table tab:general_qualities. The midfielders are part of the build-up play and are the engines on the field, but also the players expected to move the ball around and find opportunities for their teammates. Therefore qualities such as passing quality and providing teammates are used as the target variables.

| Position | Quality |
| -------- | ------- |
| Midfielder  | Passing quality|
| Midfielder | Progression|
| Midfielder    | Providing teammates|
| Midfielder| Composure|

Lastly, the defenders qualities that are used as target variables are shown in table tab:defensive_qualities. These qualities describes the defensive part of the game, where these positions aim to win the ball back in the case of an opposition attack and provide stability in front of their goalkeeper. 

| Position | Quality |
| -------- | ------- |
| Full back  | Active defence|
| Full back | Involvement|
| Full back    | Intelligent defence|
| Central Defender| Active defence|
| Central Defender| Aerial threat|
| Central Defender| Winning duels|

Continuing, the team success model will have the same layout and pipeline as the player success model. This model will also use qualities, but instead of player qualities it will use team qualities, describing how the team plays in certain areas of the field. This will be important when analyzing the affect a player has on the team and will need to be categorized according to the area which a player occupies on the field. For example, if an offensive player is inputted into the pipeline, the team model will use target variables that describe the offensive part of the field, to grab a context of whether the player will improve this part of the field.

Further, by using the qualities as a measurement of success, it does not only provide a good way of measuring how a player fits in another position but also a good way of separating different sub-positions without them being present. 

### Models

For this project, different models will be used to analyze the performance off different models in this context. The structure of the models will be similar, in terms of using the same target variable and the same independent variables to explain the problem. The different models that will be used are shown below.

* Ordinary least squares, (OLS)
* Lasso regression
* Ridge regresion
* XgBoost
* Random Forest

For this project, two different machine learning model types will be used. Interpretable models and black-box models. Interpretable models are models in which humans can understand the prediction and the models decision while for black-box models the work to the prediction is hidden behind the model. 

Continuing, for the interpretable models OLS, Lasso and ridge - regression were all used as different techniques and ways to go about having linear regression for this problem. They all provide slightly different aspects as machine learning models, as OLS does not add any penalty to its prediction, it leads to it having a simpler cost function. While both Lasso and Ridge regression are both techniques which add a penalty to the cost function.

XgBoost and Random forest were used to see how other type of models, more importantly black-box models, would perform at the task at hand. Both these algorithms are different in comparison to the regression algorithms, as they construct trees in different ways. Random forest is a more interpretable algorithm, providing a way of explaining its decision and introduces a randomness into its creation of individual decision tress. In comparison, XgBoost prunes its trees by correcting the errors each creation of a decision tree makes because of its sequential creation of decision trees. It is also a model that is slightly better for performance, and handles larger data better as it can run on multiple cores. With these black-box models, post-hoc interpretability was used to understand the models thought process behind its prediction. These post-hoc methods were added after the models prediction and will be explained further under the section Measurment.

### Naive baseline models
Continuing, a baseline model will be created to be able to compare whether the model actually improves the prediction for the problem. To provide some sort of value to the baseline model, there need to be some sort of prediction to it. Since, only using an average of the target variable or similar is not actually a good baseline as it does not provide context to the success of the transfer. As well as this, two baseline models will be used to see how adding context to the models will improve the prediction.

Therefore, the first naive linear regression model is only provided with the simplest possible context to the problem. Being the quality before transitioning to the other team. Creating a mathematical prediction as show below in the equation. 

gamma = alpha + beta * quality_{before}

This will give the simplest baseline model of how the quality actually transitions when transferring to another team, giving the other models an ability to counter the baseline model with more context to draw conclusions whether it works.

Further, the second baseline model will be provided with the context of all player qualities, establishing more context regarding the player before and after a transition but no context regarding the teams playing styles.

## Team success model

To analyze how a team might experience success based on transferring a certain player, a team success model will be created. The team success model will be different from the player success model, as it does not attribute the success as a value for different transitions, but instead uses the players ability as a predictor to the success.

### Team success
The term success in the context of team sports is usually target towards the amount of titles a team win during a season or the amount of points a team manages to get during a league season. However, this is not what this model will attempt to predict as the single effect of a player transfer is not usually something that is determinant in the outcome of a teams season.

Instead, this project will aim to see whether the team success model can predict the effect a player will have on the teams playing style, providing a way for football analysts to gather a quick overview over possible transfers effect on the teams style of play. Where analysts will be able to draw conclusions in whether the player might fit the playing style the team wants to play or whether the player might affect it in a way that is sought after from them. 

### Structure

Looking at the structure of the model, the model will structure itself around the same data as the player success model. Where the model will use the player qualities and the players current teams playing style, in order to be able to have the context of the players abilities as well as the players current environment that they are playing in. This will help in analyzing whether the effect to a teams playing style by adding a player to that team can be predicted using machine learning.

Firstly, a team can have made more than one transfer of a player during a season that has played significantly for them. This leads to the likelihood of a single one of these players affecting the outcome of the playing style very low. Therefore, a constraint on only using data containing teams that have made one significant transfer will be used. 

Further, a team that has made a significant transfer will be defined as a team that have only made one transfer of a player playing a total > 1200 minutes. This will be done in order to give the model more context into single players contribution to the teams playing style. 

Looking at the way a season is structured with most competitions having around 38 games, which has a default playing time of 90 minutes each game. The total playing time for a player playing all minutes in all games is 38 * 90 = 3420. However, having a player playing all minutes is not likely, as the way a season is structured today with other competitions and a tight playing schedule players are expected to get some rest. With that, setting a threshold at around a third of the minutes played during a league season is a good threshold for the player playing and making an impact on the team.

Contiuing, the target variables are team qualities, which describe the teams playing style during a game. These are variables created by using a weighted average sum of raw team stats, using the same principle as the player qualities. Visualized in the table below. the team qualities that will be the sought after variables for the models to predict.

| Quality |
| --------|
| Attack|
| Attacking transition|
| Defence|
| Defensive transition| 
| Chance creation|
| Penetration|

Further, an important statement made was the prediction of the effect a player has on their target teams style of play. The effect of the players entrance to a team, is determined by looking at the change of a team quality when a player has entered a team subtracted from what it was before. This gives the delta in the team quality that the model is intended to look for which can be seen below in the equation.

delta_quality = Q_prior - Q_current

In the equation above the delta is the change in quality, the Q_prior is the team quality the season prior the transfer was made and the Q_current is the quality from after the transfer was made and the player had played on the team. This will be the target variable used in the training phase of the model. For this part of the pipeline, only XgBoost will be used to predict the outcome of the variable. 

### Naive baseline model
For this problem, two baseline models will be created. Where one baseline model will be provided with the context of only using player qualities. While the other baseline model will only be provided with the context of using the players current team style of play. This will be used in the same manner as the baseline models provided for the player qualities.

### Final model
Further, the final model will use both context given into the baseline model, providing a full model containing both of the team context and the player context to give the model the full context. 

## Modeling techniques

### Feature correlation
With the data that will be used as the independent variables for each model, it exists a chance that the correlation between features might be high. Therefore a correlation step will be introduced in order to remove highly correlated attributes. 

The qualities that will be used as the independent variables of the model will explain multiple attributes as a weighted sum. This means that multiple qualities could consist of the same variables in its weighted sum, with one or two variables creating a difference for the qualities. This means that there is a high chance of certain features being highly correlated. This means that the features will provide similar information to the model, which in some cases is redundant and can provide the model with overfitting \cite{feature-correlation}. However, this step also removes features, which might remove necessary information from the model and there is also a chance that features that appear as highly correlated might provide very distinct information \cite{feature-correlation}.

Therefore a threshold has been set of $0.9$ in feature correlation, meaning that no features with a correlation of less than 0.9 will be removed. This will ensure that only extremely high correlated features will be removed and give the model a chance to improvements in its performance, since it gives the model a chance to train its values on a set of data that has a decreased amount of variables, which will speed up the training of the model and generally improves the interpret ability of the model. 


### Backwards elimination

As well as explaining the thought process behind a models outcome, its also important to explain the thought process of how features and variables was applied to a model. For this project, backwards elimination was used as a step in the training phase of the model. Backward elimination is a technique in which all features is applied to a machine learning model, then a threshold is selected for the significance level of the p-value of the feature, which is usually 0.05. The p-value for a feature is most commonly used in regression models and explains how much of importance the feature has in determining the prediction of the value. For the black-box models their feature-importances were used, were each feature with an importance of zero was removed. This is then done iteratively until their exist no feature that exceeds the significance level chosen. 

This will be done with the intent to remove features that have no affect on the model outcome, which will improve the performance of the model and it will also decrease runtime when training the different models during the training phase. It will also improve the interpretability of the model, with less features and also a model containing only the most important features it will be easier to interpret the importance of these features.

## Measurements

To analyze the models performance and results different methods will be used in order to determine this. 

### Model

For the model performance, different model metrics will be used such as $R^2$ and Mean absolute error. These metrics are commonly used in the machine learning field, as $R^2$ is a measurement of the explainability of the model and describes how much of the data the model actually explains. While Mean absolute error describes the magnitude of the difference between actual value and the predicted value during the training phase of the models. 

These performance metrics will give a good baseline into how the model performs in terms of coming into a conclusion based on the data that exists. However, these are not the only measurements that will be used as the explainability of the features affect on the model outcome is also important. 

### Interpretable coefficients
Generally, when trying to create models and pipelines of these types the interpretability of the models are of utmost importance. As the way different qualities and abilities around a player is important to the analysis of the model and to see which qualities correlate to a high value in a target quality when making a transition. 

For the interpretable models, the coefficients describing how the features affect the prediction will be used as a measurement of the effect a feature has on the prediction. 

However for the black-box models, since they do not have any coefficients, Shapley additive values, or SHAP-values, will be used. SHAP-values are single measurements of how certain features within a machine learning model affect the outcome of the models prediction. SHAP-values stem from using game theory in assigning credits to a model coefficient, depending on when the value of a feature is known or unknown \cite{shap-val}. The additive part of the SHAP-values is created by the idea that the SHAP-values of all features sum up to the models expected value \cite{shap-val}. 

However, since SHAP-values also assign the value to the features for a single prediction, as the method is dependent on the prediction of the features, this project will use the mean value of all the SHAP-values for a feature in order to make a conclusion on its importance. 

Continuing, one of the questions to answer for this project was:
Where do we fit a player in order to maximize his success?

To answer this, the results from the interpretable coefficients will be used to make this analysis. This will be done by applying an analysis pipeline, where the mean values for each of the coefficients for each model will be calculated to get an overview of the importance a feature has in a transfer transition. 

Further, then for each positional transition the average of the feature value for the three different target models will be calculated to get an average overall value of the feature for the positional transition. Lastly, the overall highest value for the three different average values for the three transition models for a position will then result in the transition in which the quality has the highest chance of succeeding in.

### Positional analysis
Since the project aims to predict whether a player could fit in a given team, and where he would be most successful at playing, an analysis method is applied to the different models outputs in order to get the position in which the player is deemed to be the most successful to play at. 

Therefore, since a player will be predicted into three different positions, one being the position they already play in and the other being the position that they currently play in, and at the same time for each transition have three predicted qualities the final output will be in total nine predicted scores differentiated on three positions. 

To get the position that the player will be most successful in, the average value of the three qualities per position will be taken as the positional value. This is done since no position requires only one type of skill in order to be successful, by taking the mean the result becomes the average over all the qualities providing a more meaningful score when deciding position. Lastly, the maximum value from the three positions is taken and the output becomes the best position for the player. The quality predictions will still be maintained, so that an analysis of the players best predicted quality in that position can be visualized. 

Overall, this gives a way of combining predicting singular qualities into a positional analysis for the players and the model, which gives more analysis into how the player would play in the other team instead of only categorizing them into a position.