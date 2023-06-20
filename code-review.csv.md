# Code Review Results

## `/scripts/DescriptiveAnalysis.py`
### Position: `47:0-55:0`
* Priority: `0`
* Title: `Add reference to math`
* Category: `Maintainability`
* Description: `Maybe add a reference to the math for computing the lower and upper bounds. It makes checking correctness easier and you may need to refer to it at a later point in time.`
* SHA: `e4ac40563dfade21ced05cf94d46d72c164830af`
### Position: `212:0-214:0`
* Priority: `0`
* Title: `Variable names too long`
* Category: `Best Practices`
* Description: `I would call it df_patternmean for example.`
* SHA: `e4ac40563dfade21ced05cf94d46d72c164830af`
### Position: `216:0-219:0`
* Priority: `0`
* Title: `Lines too long, hard to read`
* Category: `Code-Style`
* Description: `The first line is hard to parse. Part of the problem is the long variable name. But you could also split up this line into more smaller lines with intermediate variables.

It might be clearer if you used a percentile function and got the value at 33, 66 percentiles. Also there might be a function in pandas that does this binning for you.`
* SHA: `e4ac40563dfade21ced05cf94d46d72c164830af`
### Position: `221:0-228:0`
* Priority: `0`
* Title: `Just make new columns with pandas`
* Category: `Best Practices`
* Description: `You can avoid the .apply here by simply making the &quot;full symmetry&quot; and &quot;unidirection symmetry&quot; columns in the dataframe.`
* SHA: `e4ac40563dfade21ced05cf94d46d72c164830af`
### Position: `230:0-246:0`
* Priority: `0`
* Title: `Separate plotting code from data processing code`
* Category: `Code-Style`
* Description: `This block is in the same cell as the block above. I would separate the plotting code into a different cell in jupyter noteobook.`
* SHA: `e4ac40563dfade21ced05cf94d46d72c164830af`
### Position: `256:0-262:0`
* Priority: `0`
* Title: `Do you need to shuffle the train and test split?`
* Category: `Reliability`
* Description: `Are there any biases to using the first 40 rows for testing? Do you need to shuffle the test and train rows?`
* SHA: `e4ac40563dfade21ced05cf94d46d72c164830af`
## `/measures/quadtree/calculate_quadtree.py`
### Position: `47:71-47:85`
* Priority: `0`
* Title: `len(np.arange(0,grid_size_rows//2))`
* Category: `Code-Style`
* Description: `Why len(np.arange(0,grid_size_rows//2))?

Can&#x27;t you do simply set the length to (grid_size_rows//2-0)?`
* SHA: `439e8d79198575f9386a9b049829eb228cab1626`
## `/measures/local spatial complexity/calculate_local_spatial_complexity.py`
### Position: `27:0-76:0`
* Priority: `0`
* Title: `Break into a helper function`
* Category: `Best Practices`
* Description: `This should be a helper function to increase readability of the code.`
* SHA: `439e8d79198575f9386a9b049829eb228cab1626`
### Position: `29:5-32:23`
* Priority: `0`
* Title: `Simplify the logic`
* Category: `Complexity`
* Description: `Perhaps you can use a dictionary to map from dirn --&gt; (i&#x27;, j&#x27;) so that you can simply write,


if grid_size &gt; i&#x27; &gt; 0 and grid_size &gt; j&#x27; &gt; 0: 
  if grid[i&#x27;, j&#x27;] &#x3D;&#x3D; col2:
    count_neigh +&#x3D; 1
  if grid[i, j] &#x3D;&#x3D; col1 and grid[i&#x27;, j&#x27;] &#x3D;&#x3D; col2:
    count_tuples +&#x3D; 1`
* SHA: `439e8d79198575f9386a9b049829eb228cab1626`
### Position: `93:0-94:0`
* Priority: `0`
* Title: `&quot;pattern&quot; vs &quot;grid&quot;`
* Category: `Best Practices`
* Description: `The grid argument is called &quot;grid&quot; in other metric files. It helps to be consistent. Then pattern.copy() could be named &quot;grid_copy&quot;`
* SHA: `439e8d79198575f9386a9b049829eb228cab1626`
### Position: `138:0-142:0`
* Priority: `0`
* Title: `Stronger/more interesting test case`
* Category: `Reliability`
* Description: `Perhaps a stronger test case is required here. These 4 patterns don&#x27;t test local spatial complexity very extensively.`
* SHA: `439e8d79198575f9386a9b049829eb228cab1626`
### Position: `147:0-151:0`
* Priority: `0`
* Title: `Where do these test values come from?`
* Category: `Reliability`
* Description: `Why did you get the test values/true outputs of the function to test against? Is that source reliable?`
* SHA: `439e8d79198575f9386a9b049829eb228cab1626`
## `/measures/intricacy/calculate_intricacy.py`
### Position: `112:0-118:0`
* Priority: `0`
* Title: `Helper function`
* Category: `Best Practices`
* Description: `You should put this code into a helper function &quot;initialize_graph()&quot; or &quot;initialize_graph_edges()&quot; for readability. You&#x27;d call the function twice, once for 4-n and once for 8-n which means the code is slower but readability is more important since the code is very fast anyway for your grid sizes.`
* SHA: `439e8d79198575f9386a9b049829eb228cab1626`
## `/measures/entropy_of_means/calculate_entropy_of_means.py`
### Position: `30:0-42:0`
* Priority: `0`
* Title: `Why use a third party implementation?`
* Category: `Reliability`
* Description: `Why use the stackexchange solution? If I understand the metric correctly, I think you need 2 functions:
- smooth() which could use a convolution to smooth the grid
- entropy() which takes the smoothed grid and computes the entropy with sum over p*logp`
* SHA: `439e8d79198575f9386a9b049829eb228cab1626`
## `/scripts/MixedEffectsModelling.R`
### Position: `5:0-15:0`
* Priority: `0`
* Title: `Split up the script into smaller files?`
* Category: `Best Practices`
* Description: `I&#x27;m not familiar with R so I&#x27;m not what the best solution is but having a single script with all this functionality makes the code harder to understand.`
* SHA: `a35e3c807291b4ab176a5c761a57665048db2b87`
