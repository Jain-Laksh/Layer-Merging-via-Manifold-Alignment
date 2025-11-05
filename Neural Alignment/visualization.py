import plotly.graph_objects as go

# Data
compression = [3.125, 12.5, 34.375, 40.625, 46.875]
accuracy = [0.647770972795898, 0.6333855576128756, 0.6124483691781798, 0.639225181598063, 0.2631391539666714]

mka_accuracy = [0.662,0.547,0.6487,0.6342,0.300]


# Create figure
fig = go.Figure()

# Add main trace
fig.add_trace(go.Scatter(
    x=compression,
    y=accuracy,
    mode='lines+markers',
    line=dict(color='royalblue', width=3),  # straight lines
    marker=dict(size=10, color='darkblue', symbol='circle'),
    name='Accuracy after Neural Alignment',
    hovertemplate='Compression: %{x:.2f}%<br>Accuracy: %{y:.3f}<extra></extra>'
))

# Add MKA trace
fig.add_trace(go.Scatter(
    x=compression,
    y=mka_accuracy,
    mode='lines+markers',
    line=dict(color='green', width=3),  # straight lines
    marker=dict(size=10, color='darkgreen', symbol='circle'),
    name='MKA Accuracy',
    hovertemplate='Compression: %{x:.2f}%<br>Accuracy: %{y:.3f}<extra></extra>'
))

# Add optional threshold line
fig.add_shape(
    type='line',
    x0=0, x1=50,
    y0=0.6629, y1=0.6629,
    line=dict(color='red', width=2, dash='dash')
)

# Layout styling
fig.update_layout(
    title=dict(
        text="Accuracy vs Compression Ratio (Neural Alignment Method)",
        x=0.5,
        font=dict(size=22)
    ),
    xaxis=dict(
        title="Compression Ratio (%)",
        gridcolor='rgba(200,200,200,0.3)',
        zeroline=False,
        tickfont=dict(size=14)
    ),
    yaxis=dict(
        title="MMLU Accuracy",
        gridcolor='rgba(200,200,200,0.3)',
        tickfont=dict(size=14),
        range=[0.2, 0.7]
    ),
    plot_bgcolor='white',
    width=800,
    height=500,
    hovermode='x unified',
    font=dict(family="Arial", size=14)
)

fig.show()
