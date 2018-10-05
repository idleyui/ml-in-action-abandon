import matplotlib.pyplot as plt
import operator
import pickle

decisionNode = dict(boxstyle='round', pad=0.1, fc='0.8')
leafNode = dict(boxstyle='circle', fc='0.8')
arrow_args = dict(arrowstyle="<-")  # 定义箭头格式


# nodeTxt - 结点名
# centerPt - 文本位置
# parentPt - 标注的箭头位置
# nodeType - 结点格式
def plotNode(nodeTxt, parentPt, centerPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotTree(tree, f, t, level):
    level = min(level, 5)
    if isinstance(tree, float):
        plotNode(str(tree), f, t, leafNode)
    else:
        plotNode(tree['label'], f, t, decisionNode)
        plotTree(tree['left'], t, (t[0] - 0.3 / 2 ** level, t[1] - 0.1), level + 1)
        plotTree(tree['right'], t, (t[0] + 0.3 / 2 ** level, t[1] - 0.1), level + 1)


def createPlot(tree):
    fig = plt.figure(figsize=[18, 8], facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    # createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotTree.center = (0, 0)
    plotTree(tree, (0.5, 1.0), (0.5, 1.0), 0)  # 绘制决策树
    # plotNode('decision', (0.1, 0.5), (0.1, 0.5), decisionNode)
    # plotNode('leaf', (0.3, 0.8), (0.8, 0.1), leafNode)
    plt.show()


createPlot({
    'label': 'height',
    'val': 3.5,
    'data': 'Counter({1.0: 913, 2.0: 264, 5.0: 92, 4.0: 71, 3.0: 23})',
    'left': {
        'label': 'mean_tr',
        'val': 1.5,
        'data': 'Counter({2.0: 230, 1.0: 34, 4.0: 1})',
        'left': {
            'label': 'height',
            'val': 1.5,
            'data': 'Counter({1.0: 17, 2.0: 10})',
            'left': 1.0,
            'right': 1.0
        },
        'right': {
            'label': 'eccen',
            'val': 1.5,
            'data': 'Counter({2.0: 220, 1.0: 17, 4.0: 1})',
            'left': 1.0,
            'right': {
                'label': 'eccen',
                'val': 3.0,
                'data': 'Counter({2.0: 220, 1.0: 8, 4.0: 1})',
                'left': 2.0,
                'right': {
                    'label': 'mean_tr',
                    'val': 3.0,
                    'data': 'Counter({2.0: 218, 1.0: 7, 4.0: 1})',
                    'left': 2.0,
                    'right': {
                        'label': 'wb_trans',
                        'val': 1.5,
                        'data': 'Counter({2.0: 193, 1.0: 6, 4.0: 1})',
                        'left': 2.0,
                        'right': 2.0
                    }
                }
            }
        }
    },
    'right': {
        'label': 'lenght',
        'val': 3.0,
        'data': 'Counter({1.0: 879, 5.0: 92, 4.0: 70, 2.0: 34, 3.0: 23})',
        'left': {
            'label': 'mean_tr',
            'val': 4.5,
            'data': 'Counter({4.0: 65, 2.0: 3, 1.0: 2, 5.0: 2})',
            'left': 5.0,
            'right': 4.0
        },
        'right': {
            'label': 'eccen',
            'val': 4.5,
            'data': 'Counter({1.0: 877, 5.0: 90, 2.0: 31, 3.0: 23, 4.0: 5})',
            'left': {
                'label': 'wb_trans',
                'val': 4.0,
                'data': 'Counter({1.0: 249, 5.0: 86, 3.0: 23, 2.0: 7, 4.0: 5})',
                'left': {
                    'label': 'mean_tr',
                    'val': 3.5,
                    'data': 'Counter({5.0: 21, 1.0: 14, 2.0: 6})',
                    'left': {
                        'label': 'wb_trans',
                        'val': 3.0,
                        'data': 'Counter({5.0: 7, 1.0: 1})',
                        'left': 5.0,
                        'right': 5.0
                    },
                    'right': {
                        'label': 'wb_trans',
                        'val': 1.5,
                        'data': 'Counter({5.0: 14, 1.0: 13, 2.0: 6})',
                        'left': {
                            'label': 'eccen',
                            'val': 3.0,
                            'data': 'Counter({2.0: 5, 1.0: 4, 5.0: 3})',
                            'left': 1.0,
                            'right': 2.0
                        },
                        'right': {
                            'label': 'eccen',
                            'val': 3.0,
                            'data': 'Counter({5.0: 11, 1.0: 9, 2.0: 1})',
                            'left': {
                                'label': 'wb_trans',
                                'val': 3.5,
                                'data': 'Counter({5.0: 11, 1.0: 7, 2.0: 1})',
                                'left': 5.0,
                                'right': 5.0
                            },
                            'right': 1.0
                        }
                    }
                },
                'right': {
                    'label': 'mean_tr',
                    'val': 3.5,
                    'data': 'Counter({1.0: 235, 5.0: 65, 3.0: 23, 4.0: 5, 2.0: 1})',
                    'left': {
                        'label': 'eccen',
                        'val': 2.5,
                        'data': 'Counter({1.0: 108, 5.0: 41, 4.0: 3, 2.0: 1})',
                        'left': {
                            'label': 'lenght',
                            'val': 4.5,
                            'data': 'Counter({1.0: 57, 5.0: 39, 4.0: 2, 2.0: 1})',
                            'left': {
                                'label': 'mean_tr',
                                'val': 2.5,
                                'data': 'Counter({4.0: 2, 1.0: 1, 5.0: 1})',
                                'left': 1.0,
                                'right': 4.0
                            },
                            'right': {
                                'label': 'mean_tr',
                                'val': 3.0,
                                'data': 'Counter({1.0: 56, 5.0: 38, 2.0: 1})',
                                'left': {
                                    'label': 'mean_tr',
                                    'val': 1.5,
                                    'data': 'Counter({1.0: 50, 5.0: 32, 2.0: 1})',
                                    'left': 1.0,
                                    'right': 1.0
                                },
                                'right': 1.0
                            }
                        },
                        'right': {
                            'label': 'mean_tr',
                            'val': 2.5,
                            'data': 'Counter({1.0: 51, 5.0: 2, 4.0: 1})',
                            'left': {
                                'label': 'eccen',
                                'val': 3.0,
                                'data': 'Counter({1.0: 40, 4.0: 1, 5.0: 1})',
                                'left': 1.0,
                                'right': 1.0
                            },
                            'right': {
                                'label': 'eccen',
                                'val': 3.0,
                                'data': 'Counter({1.0: 11, 5.0: 1})',
                                'left': {
                                    'label': 'mean_tr',
                                    'val': 3.0,
                                    'data': 'Counter({1.0: 4, 5.0: 1})',
                                    'left': 1.0,
                                    'right': 1.0
                                },
                                'right': 1.0
                            }
                        }
                    },
                    'right': {
                        'label': 'eccen',
                        'val': 3.0,
                        'data': 'Counter({1.0: 127, 5.0: 24, 3.0: 23, 4.0: 2})',
                        'left': {
                            'label': 'mean_tr',
                            'val': 4.0,
                            'data': 'Counter({1.0: 120, 3.0: 22, 5.0: 13, 4.0: 2})',
                            'left': {
                                'label': 'eccen',
                                'val': 2.0,
                                'data': 'Counter({1.0: 4, 5.0: 3, 3.0: 1})',
                                'left': 1.0,
                                'right': 5.0
                            },
                            'right': {
                                'label': 'eccen',
                                'val': 2.5,
                                'data': 'Counter({1.0: 116, 3.0: 21, 5.0: 10, 4.0: 2})',
                                'left': {
                                    'label': 'eccen',
                                    'val': 2.0,
                                    'data': 'Counter({1.0: 110, 3.0: 17, 5.0: 10, 4.0: 2})',
                                    'left': {
                                        'label': 'mean_tr',
                                        'val': 4.5,
                                        'data': 'Counter({1.0: 100, 3.0: 17, 5.0: 10, 4.0: 2})',
                                        'left': 1.0,
                                        'right': 1.0
                                    },
                                    'right': 1.0
                                },
                                'right': 1.0
                            }
                        },
                        'right': {
                            'label': 'eccen',
                            'val': 4.0,
                            'data': 'Counter({5.0: 11, 1.0: 7, 3.0: 1})',
                            'left': 5.0,
                            'right': 5.0
                        }
                    }
                }
            },
            'right': {
                'label': 'mean_tr',
                'val': 3.5,
                'data': 'Counter({1.0: 628, 2.0: 24, 5.0: 4})',
                'left': {
                    'label': 'mean_tr',
                    'val': 1.5,
                    'data': 'Counter({1.0: 499, 2.0: 1})',
                    'left': 1.0,
                    'right': 1.0
                },
                'right': {
                    'label': 'wb_trans',
                    'val': 3.0,
                    'data': 'Counter({1.0: 129, 2.0: 23, 5.0: 4})',
                    'left': 2.0,
                    'right': 1.0
                }
            }
        }
    }
})
