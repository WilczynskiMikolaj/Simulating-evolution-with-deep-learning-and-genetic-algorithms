import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

data = pd.read_csv('SimulationStats/simulation_stats0.csv')

print(data.head())

# Plot Fitness Over Generations
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Generation'], y=data['Best Organism'], mode='lines', name='Best Fitness'))
fig.add_trace(go.Scatter(x=data['Generation'], y=data['Average fitness'], mode='lines', name='Average Fitness'))
fig.update_layout(title='Fitness Over Generations', xaxis_title='Generation', yaxis_title='Fitness')
fig.show()

# Plot Food Left Over Generations
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Generation'], y=data['Food Left'], mode='lines', name='Food Left'))
fig.update_layout(title='Food Left Over Generations', xaxis_title='Generation', yaxis_title='Food Left')
fig.show()

# Plot Speed Over Generations
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Generation'], y=data['Avg Speed'], mode='lines', name='Average Speed'))
fig.update_layout(title='Speed Over Generations', xaxis_title='Generation', yaxis_title='Speed')
fig.show()

# Plot Standard Deviation of Fitness Over Generations
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Generation'], y=data['Standard deviation Fitness'], mode='lines', name='Standard Deviation of Fitness'))
fig.update_layout(title='Standard Deviation of Fitness Over Generations', xaxis_title='Generation', yaxis_title='Standard Deviation')
fig.show()

# Display summary statistics
print(data.describe())

# Display the correlation matrix
correlation_matrix = data.corr()
fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto", color_continuous_scale='Viridis')
fig.update_layout(title='Correlation Matrix')
fig.show()