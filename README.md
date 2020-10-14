# Automating problem-setting in climbing with deep generative models.

Route-setters combine technical craft and artistic skills to reproduce real rock moves on artifical climbing walls. 
In order to both keep the customers of climbing gyms challenged and to deal with the erosion of the holds, routes need to be changed very often. With indoor climbing recent boom, the demand for route-setters is rapidly increasing.

Can we use Machine Learning to automate route-setting and reduce this demand?

As a first step towards this ambitious goal we will consider a setting where the topology of both the wall and the climbing holds is fixed, such as MoonBoards and Tension Boards. Plotted against a grid of lettered and numbered coordinates, each board hold is rotated and set in a specific location. Climbers can share problems with fellow users around the globe via a dedicated app, and with thousands of boards available in climbing gyms around the world, a large quantity of graded routes (>50k) is available online. We propose to make use of this data to automate climbing routes generation by making use of state-of-the-art Machine Learning techniques. Our Generative Model will be able to, given a user specified grade, generate a climbing route of that difficulty. Moreover users will be able to get instaneous feedback on the difficulty of human crafted routes. We believe that this approach will make training with such boards more productive, and represent a fundamental step towards fully automated climbing route generation.
