# Methods

For this project there are a lot of tools and methods to consider, since football has started to become more data driven there are lots of different data to choose from. Also, since machine learning is constantly growing the field is becoming more and more wide and new techniques are constantly forming. 


## Data

### Data types

To provide good models and also a good baseline for evaluation the data is very important in this project. The data can be categorized into two separate categories seen below.

* Event data
* Transfer data

The event data, for this project consist of both individual and team data describing actions on the field. For example Goals, Assists, team entries into the box and so on. This type of data is what the models will be built upon, setting a good base for the models. The event data will be converted into z scores for the model analysis to reducing the risk of outliers and keeping the data on a similar scale since the models will handle a lot of variables. Also, the event data will be converted into qualities, which is a description of a weighted sum of these raw stats. This will be used since it will be easier to categorize players into these qualities and is of more use as a baseline for target variables. Since it gives a more clear output regarding what the player is good at. 

The transfer data is in this case used for the training part of this project. The transfer data will contain teams, competitions, countries, positions that they played for before and after the transfer. This together with the event data provides a good baseline of the model creation.

### Positional changes
Furthermore, with the data there are also the positional changes to consider, as all the positional changes within a football pitch is redundant to analyze. 

First, on the football pitch there are 11 players which can be categorized into multiple positions. For this project, the positions that will be analyzed are the strikers, winger, midfielders, full backs and central defenders.

Therefore, during this project some positions will be grouped together. So the positions full back and winger will cover both sides of the pitch, but still describe the general player playing on the side of the pitch and either offensively or defensively. The midfielder position present in the data describes a lot of different positions within the midfield. 
Since this limitation currently exists, the player qualities can be used to describe certain abilities of a player in certain areas of a position. Defensive midfielders are players playing further down the pitch, providing the last line before the defense and therefore being a bigger part of the defensive aspect of the game and also a huge part in the build-up phase. The build-up phase is the part of football where a team tries to build up an attack from their goalkeeper or the defensive line. This means that a defensive midfielder does often need to be more attributed towards the defensive side as well as having a good passing ability to distribute the ball. 

At the same time, the attacking midfielder operates in the offensive areas of the pitch. This often results in these players providing a late charge to the attacking side of the game, where these players need more attributes that favors goal-scoring, moving into the penalty area and finding the final pass before a goal.

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

## Model preparation

In machine learning pre-processing is a very important step in the methodology, as it provides the analysis a step in which the data can be cleaned and more prepared. As mentioned above in the section about data, the data that will be used are qualities, which explains multiple features as one weighted sum. 

### Data 
In all machine learning pre-processing of the data, and more importantly the ability to handle missing values or outliers is of outmost importance. In this project missing values are something that will be run into often times, as different teams or competitions might have failed to register statistics at a certain point. 

Therefore, the pre-processing of missing values will not be to remove these rows as it would be insufficient to remove these many rows. Instead the missing values will be filled with the average value for that column based on the values that exists for that feature in that players specific position and competition. This is done in order to keep as much information as possible, while still providing some context into the players ability by grouping the calculation by position and league. Since the players ability is most likely not better than the level of the competition the player plays in aswell as no better than the peers in the same position.

### Feature correlation
With the data that will be used as the variable targets of each model, it's a high chance that the correlation between features might be high. Therefore a correlation step will be introduced in order to remove highly correlated attributes. 

The qualities that will be used as the independent variables of the model will explain multiple attributes as a weighted sum. This means that multiple qualities could consist of the same variables in its weighted sum, with one or two variables creating a difference for the qualities. This means that there is a high chance of certain features being highly correlated. This means that the features will provide similar information to the model, which in some cases is redundant and can provide the model with overfitting. However, this step also removes features, which might remove necessary information from the model and there is also a chance that features that appear as highly correlated might provide very distinct information.

Therefore a threshold has been set of $0.9$ in feature correlation, meaning that no features with a correlation of less than 0.9 will be removed. This will ensure that only extremely high correlated features will be removed and give the model a chance to improvements in its performance, since it gives the model a chance to train its values on a set of data that has a decreased amount of variables, which will speed up the training of the model and generally improves the interpret ability of the model.

# Models

For this project, different models will be used to analyze the performance off different models in this context. The structure of the models will be similar, in terms of using the same target variable and the same independent variables to explain the problem. The different models that will be used are shown below.

* Ordinary least squares, (OLS)
* Lasso regression
* Ridge regresion
* XgBoost
* Random Forest

## Model types
For this project, two different machine learning model types will be used. Interpretable models and black-box models. Interpretable models are models in which humans can understand the prediction and the models decision while for black-box models the work to the prediction is hidden behind the model. 

Continuing, for the interpretable models OLS, Lasso and ridge - regression were all used as different techniques and ways to go about having linear regression for this problem. They all provide slightly different aspects as machine learning models, as OLS does not add any penalty to its prediction, it leads to it having a simpler cost function. While both Lasso and Ridge regression are both techniques which add a penalty to the cost function.

XgBoost and Random forest were used to see how other type of models, more importantly black-box models, would perform at the task at hand. Both these algorithms are different in comparison to the regression algorithms, as they construct trees in different ways. Random forest is a more interpretable algorithm, providing a way of explaining its decision and introduces a randomness into its creation of individual decision tress. In comparison, XgBoost prunes its trees by correcting the errors each creation of a decision tree makes because of its sequential creation of decision trees. It is also a model that is slightly better for performance, and handles larger data better as it can run on multiple cores. With these black-box models, post-hoc interpretability was used to understand the models thought process behind its prediction. These post-hoc methods were added after the models prediction and will be explained further under the section Measurment.

### Backwards elimination

As well as explaining the thought process behind a models outcome, its also important to explain the thought process of how features and variables was applied to a model. For this project, backwards elimination was used as a step in the training phase of the model. Backward elimination is a technique in which all features is applied to a machine learning model, then a threshold is selected for the significance level of the p-value of the feature, which is usually 0.05. The p-value for a feature is most commonly used in regression models and explains how much of importance the feature has in determining the prediction of the value. For the black-box models their feature-importances were used, were each feature with an importance of zero was removed. This is then done iteratively until their exist no feature that exceeds the significance level chosen. 

This will be done with the intent to remove features that have no affect on the model outcome, which will improve the performance of the model and it will also decrease runtime when training the different models during the training phase. It will also improve the interpretability of the model, with less features and also a model containing only the most important features it will be easier to interpret the importance of these features.

## Measurements

To analyze the models performance and results different methods will be used in order to determine this. 

### Model

For the model performance, different model metrics will be used such as $R^2$ and Mean absolute error. These metrics are commonly used in the machine learning field, as $R^2$ is a measurement of the explainability of the model and describes how much of the data the model actually explains. While Mean absolute error describes the magnitude of the difference between actual value and the predicted value during the training phase of the models. 

These performance metrics will give a good baseline into how the model performs in terms of coming into a conclusion based on the data that exists. However, these are not the only measurements that will be used as the explainability of the features affect on the model outcome is also important. 

### Interpretable coefficients
For the interpretable models, the coefficients describing how the features affect the prediction will be used as a measurement of the effect a feature has on the prediction. These coefficients will be used to analyze where a player should play based on a given quality. 

This will be calculated by taking all of the different models parameters for different qualities and position, then taking the mean value for each of the separate predictions in order to get an average value for the coefficient. Where this value will then be compared for all the other mean values for that parameter from the different models. Where then the highest value would indicate that the given quality would fit better in that position. 

However for the black-box models, since they do not have any coefficients Shapley additive values, or SHAP-values, will be used. SHAP-values are single measurements of how certain features within a machine learning model affect the outcome of the models prediction. SHAP-values stem from using game theory in assigning credits to a model coefficient, depending on when the value of a feature is known or unknown. The additive part of the SHAP-values is created by the idea that the SHAP-values of all features sum up to the models expected value. 

However, since SHAP-values also assign the value to the features for a single prediction, as the method is dependent on the prediction of the features, this project will use the mean value of all the SHAP-values for a feature in order to make a conclusion on its importance. This will then be applied to the same type of calculation as the coefficients in the interpretable models, to gather information into where a player with a given quality should play.

### Positional analysis
Since the project aims to predict whether a player could fit in a given team, and where he would be most successful at playing, an analysis method is applied to the different models outputs in order to get the position in which the player is deemed to be the most successful to play at. 

Therefore, since a player will be predicted into three different positions, one being the position they already play in and the other being the position that they currently play in, and at the same time for each transition have three predicted qualities the final output will be in total nine predicted scores differentiated on three positions. 

To get the position that the player will be most successful in, the average value of the three qualities per position will be taken as the positional value. This is done since no position requires only one type of skill in order to be successful, by taking the mean the result becomes the average over all the qualities providing a more meaningful score when deciding position. Lastly, the maximum value from the three positions is taken and the output becomes the best position for the player. The quality predictions will still be maintained, so that an analysis of the players best predicted quality in that position can be visualized. 

Overall, this gives a way of combining predicting singular qualities into a positional analysis for the players and the model, which gives more analysis into how the player would play in the other team instead of only categorizing them into a position.