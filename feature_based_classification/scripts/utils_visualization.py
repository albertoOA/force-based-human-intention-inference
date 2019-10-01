#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import plotly.io as pio
import plotly.plotly as py
import plotly.graph_objs as go

def embedded_variables_2D_plot_(model, scatter_color, labels, y_training, y_test=None, transparent=False, inferred=None, density=True, latent_variables=[0,1]):
    """
    Description --
    · It returns a figure which includes the scatters for all the values of the latent variables for both, 
    model and predictions and the density contour of the model data points. 
    
    Arguments --
        · model - from GPy (gplvm, bgplvm, etc.)
        · scatter_color - a list of the colors to be used in the plot with as many colors as classes (ordered 
        by class IDs)
        · labels - dictionary containing the classes IDs as keys and the name of the classes as value
        · y_training - list containing the real labels for the samples of the training set
        · y_test - list containing the real labels for the samples of the test set
        · transparent - boolean to plot or not with transparent background
        · inferred - already computed predicted values of the embedded variables for a set of data (test set)
        · density - is a boolean to display or not the 2D Histogram contours of the data from the three classes
        · latent_varialbes - a list containing the latent variables to plot (e.g. we could have a model with
        several latent variables and just plot some of them)
    """

    if inferred is not None:
        # lists of predicted embedded variables (test samples)
        if hasattr(inferred[:,0], 'values'):
            x = inferred[:, latent_variables[0]].values
            y = inferred[:, latent_variables[1]].values
        else:
            x = inferred[:, latent_variables[0]].mean.values
            y = inferred[:, latent_variables[1]].mean.values

    # lists of optimized embedded variables (training samples)
    if hasattr(model, 'latent_mean'):
        x_model = model.latent_mean[:, latent_variables[0]].values
        y_model = model.latent_mean[:, latent_variables[1]].values
    elif hasattr(model, 'latent_space'):
        x_model = model.latent_space.mean[:, latent_variables[0]].values
        y_model = model.latent_space.mean[:, latent_variables[1]].values
    else: 
        x_model = None
        y_model = None
    
    plots_list = list()
    for label_id, label_name in labels.items():
        plot_x_class = list()
        plot_y_class = list()

        for i in range (0, len(y_training)):
            if (y_training[i] == label_id):
                plot_x_class.append(x_model[i])
                plot_y_class.append(y_model[i])

        scat = go.Scatter(x = plot_y_class, y = plot_x_class, name=label_name, opacity = 0.5, mode='markers', \
                                          marker = dict(size = 10, color = scatter_color[label_id-1], \
                                                        line = dict(width = 2,))) 
        plots_list.append(scat)
        
        if (density):
            dens = go.Histogram2dcontour(x = plot_y_class, y = plot_x_class, opacity = 0.5, \
                                         colorscale=[[0.0, 'rgba(0,0,0,0)'], [1.0, scatter_color[label_id-1]]], \
                                         reversescale=False, showscale=False)

            plots_list.append(dens) 
        
        if inferred is not None:
            del plot_x_class[:]
            del plot_y_class[:]
            for i in range (0, len(y_test)):
                if (y_test[i] == label_id):
                    plot_x_class.append(x[i])
                    plot_y_class.append(y[i])

            scat_pred = go.Scatter(x = plot_y_class, y = plot_x_class, name=label_name+' (pred)', mode='markers', \
                                          marker = dict(size = 10, color = scatter_color[label_id-1], \
                                                        line = dict(width = 2,))) 


            plots_list.append(scat_pred)

    if (transparent):
        layout = go.Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', \
                          title=' ',
                            xaxis=dict(
                                title='embedded variable 0',
                                titlefont=dict(
                                    family='Courier New, monospace',
                                    size=18,
                                    color='#7f7f7f'
                                )
                            ),
                            yaxis=dict(
                                title='embedded variable 1',
                                titlefont=dict(
                                    family='Courier New, monospace',
                                    size=18,
                                    color='#7f7f7f'
                                )
                            ))
    else:
        layout = go.Layout(paper_bgcolor='white', plot_bgcolor='white', \
                          title=' ',
                            xaxis=dict(
                                title='embedded variable 0',
                                titlefont=dict(
                                    family='Courier New, monospace',
                                    size=18,
                                    color='#7f7f7f'
                                )
                            ),
                            yaxis=dict(
                                title='embedded variable 1',
                                titlefont=dict(
                                    family='Courier New, monospace',
                                    size=18,
                                    color='#7f7f7f'
                                )
                            ))

    fig = go.Figure(data=plots_list, layout=layout)
    return fig


def embedded_variables_3D_plot_(model, scatter_color, labels, y_training, y_test=None, transparent=False, inferred=None, latent_variables=[0,1,2]):
    """
    Description --
    · It returns a figure which includes the scatters for all the values of the latent variables for both, 
    model and predictions and the density contour of the model data points. 
    
    Arguments --
        · model - from GPy (gplvm, bgplvm, etc.)
        · scatter_color - a list of the colors to be used in the plot with as many colors as classes (ordered 
        by class IDs)
        · labels - dictionary containing the classes IDs as keys and the name of the classes as value
        · y_training - list containing the real labels for the samples of the training set
        · y_test - list containing the real labels for the samples of the test set
        · transparent - boolean to plot or not with transparent background
        · inferred - already computed predicted values of the embedded variables for a set of data (test set)
        · latent_varialbes - a list containing the latent variables to plot (e.g. we could have a model with
        several latent variables and just plot some of them)
    """

    if inferred is not None:
        # lists of predicted embedded variables (test samples)
        if hasattr(inferred[:,0], 'values'):
            x = inferred[:, latent_variables[0]].values
            y = inferred[:, latent_variables[1]].values
            z = inferred[:, latent_variables[2]].values
        else:
            x = inferred[:, latent_variables[0]].mean.values
            y = inferred[:, latent_variables[1]].mean.values
            z = inferred[:, latent_variables[2]].mean.values

    # lists of optimized embedded variables (training samples)
    if hasattr(model, 'latent_mean'):
        x_model = model.latent_mean[:, latent_variables[0]].values
        y_model = model.latent_mean[:, latent_variables[1]].values
        z_model = model.latent_mean[:, latent_variables[2]].values
    elif hasattr(model, 'latent_space'):
        x_model = model.latent_space.mean[:, latent_variables[0]].values
        y_model = model.latent_space.mean[:, latent_variables[1]].values
        z_model = model.latent_space.mean[:, latent_variables[2]].values
    else: 
        x_model = None
        y_model = None
        z_model = None
    
    plots_list = list()
    for label_id, label_name in labels.items():
        plot_x_class = list()
        plot_y_class = list()
        plot_z_class = list()

        for i in range (0, len(y_training)):
            if (y_training[i] == label_id):
                plot_x_class.append(x_model[i])
                plot_y_class.append(y_model[i])
                plot_z_class.append(z_model[i])

        scat = go.Scatter3d(x = plot_y_class, y = plot_x_class, z = plot_z_class, name=label_name, opacity = 0.5, \
                            mode='markers', marker = dict(size = 10, color = scatter_color[label_id-1], \
                                                          line = dict(width = 2,)))
        
        plots_list.append(scat)
        
        
        if inferred is not None:
            del plot_x_class[:]
            del plot_y_class[:]
            del plot_z_class[:]
            for i in range (0, len(y_test)):
                if (y_test[i] == label_id):
                    plot_x_class.append(x[i])
                    plot_y_class.append(y[i])
                    plot_z_class.append(z[i])

            scat_pred = go.Scatter3d(x = plot_y_class, y = plot_x_class, z = plot_z_class, \
                                   name=label_name+' (pred)', mode='markers', \
                                   marker = dict(size = 10, color = scatter_color[label_id-1], \
                                                 line = dict(width = 2,))) 


            plots_list.append(scat_pred)

    if (transparent):
        layout = go.Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', \
                          title=' ',
                           scene=dict(
                            xaxis=dict(
                                title='embedded variable '+str(latent_variables[0]),
                                titlefont=dict(
                                    family='Courier New, monospace',
                                    size=18,
                                    color='#7f7f7f'
                                )
                            ),
                            yaxis=dict(
                                title='embedded variable '+str(latent_variables[1]),
                                titlefont=dict(
                                    family='Courier New, monospace',
                                    size=18,
                                    color='#7f7f7f'
                                )
                            ),
                            zaxis=dict(
                                title='embedded variable '+str(latent_variables[2]),
                                titlefont=dict(
                                    family='Courier New, monospace',
                                    size=18,
                                    color='#7f7f7f'
                                )
                            ))
                          )
    else:
        layout = go.Layout(paper_bgcolor='white', plot_bgcolor='white', \
                          title=' ', 
                           scene=dict(
                            xaxis=dict(
                                title='embedded variable '+str(latent_variables[0]),
                                titlefont=dict(
                                    family='Courier New, monospace',
                                    size=18,
                                    color='#7f7f7f'
                                )
                            ),
                            yaxis=dict(
                                title='embedded variable '+str(latent_variables[1]),
                                titlefont=dict(
                                    family='Courier New, monospace',
                                    size=18,
                                    color='#7f7f7f'
                                )
                            ),
                            zaxis=dict(
                                title='embedded variable '+str(latent_variables[2]),
                                titlefont=dict(
                                    family='Courier New, monospace',
                                    size=18,
                                    color='#7f7f7f'
                                )
                            ))
                          )

    fig = go.Figure(data=plots_list, layout=layout)
    return fig

def embedded_variables_3D_plot_PCA_(model, scatter_color, labels, y_training, y_test=None, transparent=False, inferred=None, latent_variables=[0,1,2]):
    """
    Description --
    · It returns a figure which includes the scatters for all the values of the latent variables for both, 
    model and predictions and the density contour of the model data points. 
    
    Arguments --
        · model - from GPy (gplvm, bgplvm, etc.)
        · scatter_color - a list of the colors to be used in the plot with as many colors as classes (ordered 
        by class IDs)
        · labels - dictionary containing the classes IDs as keys and the name of the classes as value
        · y_training - list containing the real labels for the samples of the training set
        · y_test - list containing the real labels for the samples of the test set
        · transparent - boolean to plot or not with transparent background
        · inferred - already computed predicted values of the embedded variables for a set of data (test set)
        · latent_varialbes - a list containing the latent variables to plot (e.g. we could have a model with
        several latent variables and just plot some of them)
    """

    if inferred is not None:
        # lists of predicted embedded variables (test samples)
        if hasattr(inferred[:,0], 'values'):
            x = inferred[:, latent_variables[0]].values
            y = inferred[:, latent_variables[1]].values
            z = inferred[:, latent_variables[2]].values
        else:
            x = inferred[:, latent_variables[0]].mean.values
            y = inferred[:, latent_variables[1]].mean.values
            z = inferred[:, latent_variables[2]].mean.values

    # lists of optimized embedded variables (training samples)
    if hasattr(model, 'latent_mean'):
        x_model = model.latent_mean[:, latent_variables[0]].values
        y_model = model.latent_mean[:, latent_variables[1]].values
        z_model = model.latent_mean[:, latent_variables[2]].values
    elif hasattr(model, 'latent_space'):
        x_model = model.latent_space.mean[:, latent_variables[0]].values
        y_model = model.latent_space.mean[:, latent_variables[1]].values
        z_model = model.latent_space.mean[:, latent_variables[2]].values
    else: 
        x_model = model[:,latent_variables[0]]
        y_model = model[:,latent_variables[1]]
        z_model = model[:,latent_variables[2]]
    
    plots_list = list()
    for label_id, label_name in labels.items():
        plot_x_class = list()
        plot_y_class = list()
        plot_z_class = list()

        for i in range (0, len(y_training)):
            if (y_training[i] == label_id):
                plot_x_class.append(x_model[i])
                plot_y_class.append(y_model[i])
                plot_z_class.append(z_model[i])

        scat = go.Scatter3d(x = plot_y_class, y = plot_x_class, z = plot_z_class, name=label_name, opacity = 0.5, \
                            mode='markers', marker = dict(size = 10, color = scatter_color[label_id-1], \
                                                          line = dict(width = 2,)))
        
        plots_list.append(scat)
        
        
        if inferred is not None:
            del plot_x_class[:]
            del plot_y_class[:]
            del plot_z_class[:]
            for i in range (0, len(y_test)):
                if (y_test[i] == label_id):
                    plot_x_class.append(x[i])
                    plot_y_class.append(y[i])
                    plot_z_class.append(z[i])

            scat_pred = go.Scatter3d(x = plot_y_class, y = plot_x_class, z = plot_z_class, \
                                   name=label_name+' (pred)', mode='markers', \
                                   marker = dict(size = 10, color = scatter_color[label_id-1], \
                                                 line = dict(width = 2,))) 


            plots_list.append(scat_pred)

    if (transparent):
        layout = go.Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', \
                          title=' ',
                           scene=dict(
                            xaxis=dict(
                                title='embedded variable '+str(latent_variables[0]),
                                titlefont=dict(
                                    family='Courier New, monospace',
                                    size=18,
                                    color='#7f7f7f'
                                )
                            ),
                            yaxis=dict(
                                title='embedded variable '+str(latent_variables[1]),
                                titlefont=dict(
                                    family='Courier New, monospace',
                                    size=18,
                                    color='#7f7f7f'
                                )
                            ),
                            zaxis=dict(
                                title='embedded variable '+str(latent_variables[2]),
                                titlefont=dict(
                                    family='Courier New, monospace',
                                    size=18,
                                    color='#7f7f7f'
                                )
                            ))
                          )
    else:
        layout = go.Layout(paper_bgcolor='white', plot_bgcolor='white', \
                          title=' ', 
                           scene=dict(
                            xaxis=dict(
                                title='embedded variable '+str(latent_variables[0]),
                                titlefont=dict(
                                    family='Courier New, monospace',
                                    size=18,
                                    color='#7f7f7f'
                                )
                            ),
                            yaxis=dict(
                                title='embedded variable '+str(latent_variables[1]),
                                titlefont=dict(
                                    family='Courier New, monospace',
                                    size=18,
                                    color='#7f7f7f'
                                )
                            ),
                            zaxis=dict(
                                title='embedded variable '+str(latent_variables[2]),
                                titlefont=dict(
                                    family='Courier New, monospace',
                                    size=18,
                                    color='#7f7f7f'
                                )
                            ))
                          )

    fig = go.Figure(data=plots_list, layout=layout)
    return fig
