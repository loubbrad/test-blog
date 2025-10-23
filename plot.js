// Wait for the DOM to be fully loaded before running the plot code
document.addEventListener('DOMContentLoaded', function() {

    // 1. Get references to our HTML elements
    const plotDiv = document.getElementById('tsnePlot');
    // CHANGED: Get a reference to the new mute toggle
    const muteToggle = document.getElementById('muteToggle');

    let currentAudio = null;
    let isMuted = false; // ADDED: State variable to track mute status

    // 2. Load the data from your JSON file
    fetch('assets/tsne.json')
        .then(response => response.json())
        .then(data => {
            // Data is now loaded
            console.log("Data loaded:", data);

            // 3. Process the data for Plotly
            const allClusterLabels = data.map(point => point.cluster);
            const uniqueClusters = [...new Set(allClusterLabels)];

            // Create a mapping from cluster names to numbers (for coloring)
            const clusterToNumber = {};
            uniqueClusters.forEach((cluster, index) => {
                clusterToNumber[cluster] = index;
            });

            // Define the color palette
            const tab20 = [
              '#1f77b4', '#aec7e8',
              '#ff7f0e', '#ffbb78',
              '#2ca02c', '#98df8a',
              '#d62728', '#ff9896',
              '#9467bd', '#c5b0d5',
              '#8c564b', '#c49c94',
              '#e377c2', '#f7b6d2',
              '#7f7f7f', '#c7c7c7',
              '#bcbd22', '#dbdb8d',
              '#17becf', '#9edae5'
            ];

            // 4. Create one trace per cluster (to assign colors)
            const traces = [];
            for (const clusterName of uniqueClusters) {
                
                const clusterData = data.filter(point => point.cluster === clusterName);
                const clusterIndex = clusterToNumber[clusterName];
                const clusterColor = tab20[clusterIndex % tab20.length];

                const trace = {
                    x: clusterData.map(point => point.x),
                    y: clusterData.map(point => point.y),
                    text: clusterData.map(point => point.piece),
                    customdata: clusterData.map(point => point.audioFile),
                    type: 'scattergl',
                    mode: 'markers',
                    hoverinfo: 'text',
                    marker: {
                        color: clusterColor,
                        size: 8,
                        opacity: 0.8
                    }
                };
                traces.push(trace);
            }
            
            // 5. Define the layout for the plot
            const layout = {
                title: '',
                hovermode: 'closest',
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                dragmode: false,
                xaxis: {
                    title: '',
                    showgrid: false,
                    zeroline: false,
                    showline: false,
                    showticklabels: false,
                    ticks: ''
                },
                yaxis: {
                    title: '',
                    showgrid: false,
                    zeroline: false,
                    showline: false,
                    showticklabels: false,
                    ticks: ''
                },
                showlegend: false,
                margin: { l: 0, r: 0, b: 0, t: 0 }
            };

            const config = {
                displayModeBar: false,
                scrollZoom: false,
                doubleClick: false
            };

            Plotly.newPlot(plotDiv, traces, layout, config);

            plotDiv.on('plotly_click', function(data) {
                const pointInfo = data.points[0];
                const audioSrc = 'assets/' + pointInfo.customdata; 

                console.log('Playing audio:', audioSrc);

                if (currentAudio) {
                    currentAudio.pause();
                    currentAudio.currentTime = 0;
                }

                currentAudio = new Audio(audioSrc);
                
                // ADDED: Apply the current mute state to the new audio
                currentAudio.muted = isMuted;
                
                currentAudio.play();
            });

            // CHANGED: Event listener for the mute toggle
            muteToggle.addEventListener('click', function() {
                // Toggle the mute state
                isMuted = !isMuted;
                
                // Toggle the CSS class to show/hide the line-through
                muteToggle.classList.toggle('muted', isMuted);
                
                // If audio is currently playing, apply the mute state immediately
                if (currentAudio) {
                    currentAudio.muted = isMuted;
                }
                console.log('Audio muted:', isMuted);
            });

        })
        .catch(error => console.error('Error loading data:', error));
});