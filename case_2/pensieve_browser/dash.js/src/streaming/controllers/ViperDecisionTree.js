var DecisionTreeClassifier = function() {

    var findMax = function(nums) {
        var index = 0;
        for (var i = 0; i < nums.length; i++) {
            index = nums[i] > nums[index] ? i : index;
        }
        return index;
    };

    this.predict = function(features) {
        var classes = new Array(6);
            
        if (features[0] <= 0.226744189858) {
            if (features[0] <= 0.12209302187) {
                if (features[1] <= 1.43548971415) {
                    if (features[9] <= 0.111436057836) {
                        if (features[1] <= 1.31060123444) {
                            if (features[16] <= 0.120652284473) {
                                if (features[1] <= 1.15815412998) {
                                    classes[0] = 63; 
                                    classes[1] = 0; 
                                    classes[2] = 0; 
                                    classes[3] = 0; 
                                    classes[4] = 0; 
                                    classes[5] = 0; 
                                } else {
                                    if (features[17] <= 0.164240263402) {
                                    } else {
                                    }
                                }
                            } else {
                                if (features[1] <= 1.16800129414) {
                                } else {
                                }
                            }
                        } else {
                            if (features[9] <= 0.0816933065653) {
                                if (features[5] <= 0.104554716498) {
                                    if (features[21] <= 0.915657997131) {
                                    } else {
                                    }
                                } else {
                                    if (features[1] <= 1.36486339569) {
                                        classes[0] = 31; 
                                        classes[1] = 0; 
                                        classes[2] = 0; 
                                        classes[3] = 0; 
                                        classes[4] = 0; 
                                        classes[5] = 0; 
                                    } else {
                                        classes[0] = 0; 
                                        classes[1] = 40; 
                                        classes[2] = 0; 
                                        classes[3] = 0; 
                                        classes[4] = 0; 
                                        classes[5] = 0; 
                                    }
                                }
                            } else {
                                if (features[5] <= 0.0750453658402) {
                                    if (features[4] <= 0.0854021459818) {
                                        classes[0] = 206; 
                                        classes[1] = 0; 
                                        classes[2] = 0; 
                                        classes[3] = 0; 
                                        classes[4] = 0; 
                                        classes[5] = 0; 
                                    } else {
                                        if (features[16] <= 0.176825068891) {
                                        } else {
                                        }
                                    }
                                } else {
                                    if (features[9] <= 0.0929700024426) {
                                    } else {
                                    }
                                }
                            }
                        }
                    } else {
                        if (features[1] <= 1.18146294355) {
                            if (features[17] <= 0.0879879519343) {
                                if (features[16] <= 0.267582848668) {
                                } else {
                                }
                            } else {
                                if (features[1] <= 0.980232834816) {
                                    if (features[15] <= 1.15472853184) {
                                    } else {
                                    }
                                } else {
                                    if (features[6] <= 0.195790760219) {
                                        if (features[16] <= 0.102989852428) {
                                            if (features[9] <= 0.128266297281) {
                                                classes[0] = 27; 
                                                classes[1] = 0; 
                                                classes[2] = 0; 
                                                classes[3] = 0; 
                                                classes[4] = 0; 
                                                classes[5] = 0; 
                                            } else {
                                                if (features[15] <= 0.236742570996) {
                                                } else {
                                                }
                                            }
                                        } else {
                                            if (features[15] <= 0.181544959545) {
                                            } else {
                                            }
                                        }
                                    } else {
                                        classes[0] = 0; 
                                        classes[1] = 70; 
                                        classes[2] = 0; 
                                        classes[3] = 0; 
                                        classes[4] = 0; 
                                        classes[5] = 0; 
                                    }
                                }
                            }
                        } else {
                            if (features[11] <= 0.789146125317) {
                                if (features[17] <= 0.152378357947) {
                                } else {
                                }
                            } else {
                                if (features[2] <= 0.121582698077) {
                                } else {
                                }
                            }
                        }
                    }
                } else {
                    if (features[1] <= 1.50938534737) {
                        if (features[9] <= 0.0716493427753) {
                            if (features[14] <= 0.2591868788) {
                                if (features[13] <= 0.35128518939) {
                                    if (features[17] <= 0.229289494455) {
                                        classes[0] = 42; 
                                        classes[1] = 0; 
                                        classes[2] = 0; 
                                        classes[3] = 0; 
                                        classes[4] = 0; 
                                        classes[5] = 0; 
                                    } else {
                                        if (features[8] <= 0.0658459737897) {
                                        } else {
                                        }
                                    }
                                } else {
                                    if (features[2] <= 0.0696496851742) {
                                        if (features[13] <= 0.732003092766) {
                                        } else {
                                        }
                                    } else {
                                        if (features[10] <= 0.168976821005) {
                                        } else {
                                        }
                                    }
                                }
                            } else {
                                if (features[18] <= 0.14267449826) {
                                    if (features[12] <= 0.292889028788) {
                                        classes[0] = 0; 
                                        classes[1] = 48; 
                                        classes[2] = 0; 
                                        classes[3] = 0; 
                                        classes[4] = 0; 
                                        classes[5] = 0; 
                                    } else {
                                        if (features[6] <= 0.0489143300802) {
                                            classes[0] = 47; 
                                            classes[1] = 0; 
                                            classes[2] = 0; 
                                            classes[3] = 0; 
                                            classes[4] = 0; 
                                            classes[5] = 0; 
                                        } else {
                                            if (features[11] <= 0.228810139) {
                                                if (features[1] <= 1.46896255016) {
                                                } else {
                                                }
                                            } else {
                                                classes[0] = 0; 
                                                classes[1] = 38; 
                                                classes[2] = 0; 
                                                classes[3] = 0; 
                                                classes[4] = 0; 
                                                classes[5] = 0; 
                                            }
                                        }
                                    }
                                } else {
                                    if (features[1] <= 1.43712246418) {
                                        classes[0] = 0; 
                                        classes[1] = 21; 
                                        classes[2] = 0; 
                                        classes[3] = 0; 
                                        classes[4] = 0; 
                                        classes[5] = 0; 
                                    } else {
                                        if (features[14] <= 0.690146535635) {
                                            if (features[2] <= 0.0310105290264) {
                                            } else {
                                            }
                                        } else {
                                            if (features[9] <= 0.0618184935302) {
                                                if (features[9] <= 0.0555907152593) {
                                                } else {
                                                }
                                            } else {
                                                classes[0] = 33; 
                                                classes[1] = 0; 
                                                classes[2] = 0; 
                                                classes[3] = 0; 
                                                classes[4] = 0; 
                                                classes[5] = 0; 
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if (features[9] <= 0.0966454297304) {
                                if (features[11] <= 0.29376591742) {
                                    if (features[24] <= 0.78125) {
                                        if (features[14] <= 0.828183829784) {
                                        } else {
                                        }
                                    } else {
                                        if (features[12] <= 0.546498715878) {
                                            classes[0] = 0; 
                                            classes[1] = 55; 
                                            classes[2] = 0; 
                                            classes[3] = 0; 
                                            classes[4] = 0; 
                                            classes[5] = 0; 
                                        } else {
                                            if (features[1] <= 1.49315530062) {
                                            } else {
                                            }
                                        }
                                    }
                                } else {
                                    if (features[1] <= 1.5073197484) {
                                        if (features[2] <= 0.299527183175) {
                                            if (features[3] <= 0.035061115399) {
                                            } else {
                                            }
                                        } else {
                                            classes[0] = 19; 
                                            classes[1] = 0; 
                                            classes[2] = 0; 
                                            classes[3] = 0; 
                                            classes[4] = 0; 
                                            classes[5] = 0; 
                                        }
                                    } else {
                                        classes[0] = 24; 
                                        classes[1] = 0; 
                                        classes[2] = 0; 
                                        classes[3] = 0; 
                                        classes[4] = 0; 
                                        classes[5] = 0; 
                                    }
                                }
                            } else {
                                classes[0] = 0; 
                                classes[1] = 389; 
                                classes[2] = 0; 
                                classes[3] = 0; 
                                classes[4] = 0; 
                                classes[5] = 0; 
                            }
                        }
                    } else {
                        if (features[9] <= 0.0587352793664) {
                            if (features[1] <= 1.56142526865) {
                                if (features[22] <= 1.39301097393) {
                                    if (features[24] <= 0.0312500009313) {
                                    } else {
                                    }
                                } else {
                                    if (features[1] <= 1.51818799973) {
                                        classes[0] = 0; 
                                        classes[1] = 29; 
                                        classes[2] = 0; 
                                        classes[3] = 0; 
                                        classes[4] = 0; 
                                        classes[5] = 0; 
                                    } else {
                                        if (features[24] <= 0.510416671634) {
                                        } else {
                                        }
                                    }
                                }
                            } else {
                                if (features[5] <= 0.0343108922243) {
                                } else {
                                }
                            }
                        } else {
                            if (features[17] <= 0.0411258526146) {
                            } else {
                            }
                        }
                    }
                }
            } else {
                if (features[1] <= 1.06396728754) {
                    if (features[17] <= 0.195359304547) {
                        if (features[8] <= 0.387035772204) {
                            if (features[17] <= 0.182199314237) {
                            } else {
                            }
                        } else {
                            if (features[1] <= 0.717764496803) {
                                if (features[8] <= 0.399657085538) {
                                } else {
                                }
                            } else {
                                classes[0] = 0; 
                                classes[1] = 0; 
                                classes[2] = 0; 
                                classes[3] = 106; 
                                classes[4] = 0; 
                                classes[5] = 0; 
                            }
                        }
                    } else {
                        if (features[8] <= 0.11015670374) {
                            if (features[1] <= 0.992121756077) {
                                if (features[5] <= 0.132041677833) {
                                    if (features[3] <= 0.26876090467) {
                                    } else {
                                    }
                                } else {
                                    if (features[9] <= 0.118650846183) {
                                        if (features[1] <= 0.870508730412) {
                                        } else {
                                        }
                                    } else {
                                        if (features[16] <= 0.366599157453) {
                                        } else {
                                        }
                                    }
                                }
                            } else {
                                if (features[4] <= 0.0699569061399) {
                                    if (features[10] <= 0.374333187938) {
                                    } else {
                                    }
                                } else {
                                    if (features[10] <= 0.159760840237) {
                                        classes[0] = 91; 
                                        classes[1] = 0; 
                                        classes[2] = 0; 
                                        classes[3] = 0; 
                                        classes[4] = 0; 
                                        classes[5] = 0; 
                                    } else {
                                        if (features[7] <= 0.0773128159344) {
                                            if (features[13] <= 0.249287500978) {
                                                classes[0] = 0; 
                                                classes[1] = 41; 
                                                classes[2] = 0; 
                                                classes[3] = 0; 
                                                classes[4] = 0; 
                                                classes[5] = 0; 
                                            } else {
                                                if (features[16] <= 0.401607692242) {
                                                    classes[0] = 0; 
                                                    classes[1] = 24; 
                                                    classes[2] = 0; 
                                                    classes[3] = 0; 
                                                    classes[4] = 0; 
                                                    classes[5] = 0; 
                                                } else {
                                                    if (features[20] <= 0.648961514235) {
                                                    } else {
                                                    }
                                                }
                                            }
                                        } else {
                                            if (features[23] <= 2.28106856346) {
                                                if (features[9] <= 0.0487197171897) {
                                                } else {
                                                }
                                            } else {
                                                if (features[12] <= 0.396136164665) {
                                                    classes[0] = 44; 
                                                    classes[1] = 0; 
                                                    classes[2] = 0; 
                                                    classes[3] = 0; 
                                                    classes[4] = 0; 
                                                    classes[5] = 0; 
                                                } else {
                                                    classes[0] = 0; 
                                                    classes[1] = 24; 
                                                    classes[2] = 0; 
                                                    classes[3] = 0; 
                                                    classes[4] = 0; 
                                                    classes[5] = 0; 
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if (features[1] <= 0.776594609022) {
                                if (features[8] <= 0.222049050033) {
                                    if (features[16] <= 0.783837527037) {
                                    } else {
                                    }
                                } else {
                                    classes[0] = 0; 
                                    classes[1] = 27; 
                                    classes[2] = 0; 
                                    classes[3] = 0; 
                                    classes[4] = 0; 
                                    classes[5] = 0; 
                                }
                            } else {
                                if (features[9] <= 0.0502119082958) {
                                } else {
                                }
                            }
                        }
                    }
                } else {
                    if (features[1] <= 2.30335855484) {
                        if (features[9] <= 0.217382408679) {
                            if (features[9] <= 0.0561340972781) {
                                if (features[1] <= 1.27243000269) {
                                    if (features[17] <= 0.74425688386) {
                                        if (features[15] <= 0.306910663843) {
                                            if (features[10] <= 0.54191327095) {
                                                if (features[1] <= 1.25293397903) {
                                                } else {
                                                }
                                            } else {
                                                if (features[9] <= 0.0550672691315) {
                                                } else {
                                                }
                                            }
                                        } else {
                                            if (features[4] <= 0.0597696118057) {
                                                if (features[14] <= 0.263580076396) {
                                                    classes[0] = 0; 
                                                    classes[1] = 20; 
                                                    classes[2] = 0; 
                                                    classes[3] = 0; 
                                                    classes[4] = 0; 
                                                    classes[5] = 0; 
                                                } else {
                                                    classes[0] = 54; 
                                                    classes[1] = 0; 
                                                    classes[2] = 0; 
                                                    classes[3] = 0; 
                                                    classes[4] = 0; 
                                                    classes[5] = 0; 
                                                }
                                            } else {
                                                classes[0] = 0; 
                                                classes[1] = 126; 
                                                classes[2] = 0; 
                                                classes[3] = 0; 
                                                classes[4] = 0; 
                                                classes[5] = 0; 
                                            }
                                        }
                                    } else {
                                        if (features[11] <= 0.673178851604) {
                                        } else {
                                        }
                                    }
                                } else {
                                    if (features[6] <= 0.24360113591) {
                                    } else {
                                    }
                                }
                            } else {
                                if (features[9] <= 0.175792299211) {
                                    if (features[1] <= 1.17716538906) {
                                        if (features[17] <= 0.482949957252) {
                                            if (features[12] <= 0.72555360198) {
                                                if (features[7] <= 0.0479043778032) {
                                                } else {
                                                }
                                            } else {
                                                if (features[15] <= 0.694621771574) {
                                                    if (features[13] <= 1.05393514037) {
                                                    } else {
                                                    }
                                                } else {
                                                    classes[0] = 51; 
                                                    classes[1] = 0; 
                                                    classes[2] = 0; 
                                                    classes[3] = 0; 
                                                    classes[4] = 0; 
                                                    classes[5] = 0; 
                                                }
                                            }
                                        } else {
                                            if (features[12] <= 0.584418863058) {
                                                if (features[9] <= 0.0629619881511) {
                                                    if (features[10] <= 0.271616697311) {
                                                        if (features[3] <= 0.103658061475) {
                                                            if (features[12] <= 0.568716734648) {
                                                            } else {
                                                            }
                                                        } else {
                                                            classes[0] = 0; 
                                                            classes[1] = 21; 
                                                            classes[2] = 0; 
                                                            classes[3] = 0; 
                                                            classes[4] = 0; 
                                                            classes[5] = 0; 
                                                        }
                                                    } else {
                                                        if (features[9] <= 0.0627437643707) {
                                                        } else {
                                                        }
                                                    }
                                                } else {
                                                    if (features[17] <= 0.486550435424) {
                                                    } else {
                                                    }
                                                }
                                            } else {
                                                if (features[14] <= 0.425322338939) {
                                                    if (features[3] <= 0.0543526951224) {
                                                    } else {
                                                    }
                                                } else {
                                                    classes[0] = 0; 
                                                    classes[1] = 22; 
                                                    classes[2] = 0; 
                                                    classes[3] = 0; 
                                                    classes[4] = 0; 
                                                    classes[5] = 0; 
                                                }
                                            }
                                        }
                                    } else {
                                        if (features[1] <= 2.00705552101) {
                                            if (features[9] <= 0.0711513571441) {
                                            } else {
                                            }
                                        } else {
                                            if (features[17] <= 0.238512083888) {
                                                if (features[11] <= 0.791561335325) {
                                                    if (features[1] <= 2.28428936005) {
                                                        if (features[13] <= 0.642250448465) {
                                                            if (features[9] <= 0.162934802473) {
                                                                if (features[17] <= 0.236008055508) {
                                                                    if (features[3] <= 0.0777549557388) {
                                                                    } else {
                                                                    }
                                                                } else {
                                                                    if (features[11] <= 0.28656475246) {
                                                                        classes[0] = 0; 
                                                                        classes[1] = 45; 
                                                                        classes[2] = 0; 
                                                                        classes[3] = 0; 
                                                                        classes[4] = 0; 
                                                                        classes[5] = 0; 
                                                                    } else {
                                                                        classes[0] = 0; 
                                                                        classes[1] = 0; 
                                                                        classes[2] = 0; 
                                                                        classes[3] = 38; 
                                                                        classes[4] = 0; 
                                                                        classes[5] = 0; 
                                                                    }
                                                                }
                                                            } else {
                                                                if (features[9] <= 0.163323514163) {
                                                                    if (features[24] <= 0.708333343267) {
                                                                    } else {
                                                                    }
                                                                } else {
                                                                    if (features[1] <= 2.17179477215) {
                                                                        if (features[16] <= 0.191001221538) {
                                                                        } else {
                                                                        }
                                                                    } else {
                                                                        if (features[6] <= 0.155531287193) {
                                                                            if (features[3] <= 0.121551442891) {
                                                                                classes[0] = 0; 
                                                                                classes[1] = 0; 
                                                                                classes[2] = 0; 
                                                                                classes[3] = 19; 
                                                                                classes[4] = 0; 
                                                                                classes[5] = 0; 
                                                                            } else {
                                                                                classes[0] = 0; 
                                                                                classes[1] = 66; 
                                                                                classes[2] = 0; 
                                                                                classes[3] = 0; 
                                                                                classes[4] = 0; 
                                                                                classes[5] = 0; 
                                                                            }
                                                                        } else {
                                                                            classes[0] = 0; 
                                                                            classes[1] = 0; 
                                                                            classes[2] = 0; 
                                                                            classes[3] = 48; 
                                                                            classes[4] = 0; 
                                                                            classes[5] = 0; 
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        } else {
                                                            classes[0] = 0; 
                                                            classes[1] = 0; 
                                                            classes[2] = 0; 
                                                            classes[3] = 22; 
                                                            classes[4] = 0; 
                                                            classes[5] = 0; 
                                                        }
                                                    } else {
                                                        classes[0] = 0; 
                                                        classes[1] = 0; 
                                                        classes[2] = 0; 
                                                        classes[3] = 27; 
                                                        classes[4] = 0; 
                                                        classes[5] = 0; 
                                                    }
                                                } else {
                                                    classes[0] = 0; 
                                                    classes[1] = 0; 
                                                    classes[2] = 0; 
                                                    classes[3] = 29; 
                                                    classes[4] = 0; 
                                                    classes[5] = 0; 
                                                }
                                            } else {
                                                if (features[14] <= 0.854713439941) {
                                                } else {
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    if (features[1] <= 1.85227888823) {
                                        if (features[1] <= 1.68529230356) {
                                            if (features[6] <= 0.02433286421) {
                                            } else {
                                            }
                                        } else {
                                            if (features[12] <= 0.106631245464) {
                                                if (features[17] <= 0.218655489385) {
                                                } else {
                                                }
                                            } else {
                                                if (features[10] <= 0.90521928668) {
                                                } else {
                                                }
                                            }
                                        }
                                    } else {
                                        if (features[5] <= 0.122007686645) {
                                            if (features[10] <= 0.68279248476) {
                                                if (features[1] <= 2.23236703873) {
                                                    if (features[15] <= 0.260892868042) {
                                                    } else {
                                                    }
                                                } else {
                                                    if (features[7] <= 0.160093136132) {
                                                        classes[0] = 0; 
                                                        classes[1] = 0; 
                                                        classes[2] = 0; 
                                                        classes[3] = 43; 
                                                        classes[4] = 0; 
                                                        classes[5] = 0; 
                                                    } else {
                                                        classes[0] = 0; 
                                                        classes[1] = 23; 
                                                        classes[2] = 0; 
                                                        classes[3] = 0; 
                                                        classes[4] = 0; 
                                                        classes[5] = 0; 
                                                    }
                                                }
                                            } else {
                                                classes[0] = 0; 
                                                classes[1] = 0; 
                                                classes[2] = 0; 
                                                classes[3] = 46; 
                                                classes[4] = 0; 
                                                classes[5] = 0; 
                                            }
                                        } else {
                                            if (features[7] <= 0.128583066165) {
                                                if (features[1] <= 2.03813838959) {
                                                } else {
                                                }
                                            } else {
                                                if (features[22] <= 1.55441749096) {
                                                    if (features[2] <= 0.257098689675) {
                                                        if (features[11] <= 0.0797388292849) {
                                                            if (features[11] <= 0.069821447134) {
                                                            } else {
                                                            }
                                                        } else {
                                                            if (features[4] <= 0.0851929336786) {
                                                                if (features[8] <= 0.202736735344) {
                                                                } else {
                                                                }
                                                            } else {
                                                                if (features[4] <= 0.141855634749) {
                                                                    classes[0] = 0; 
                                                                    classes[1] = 0; 
                                                                    classes[2] = 0; 
                                                                    classes[3] = 165; 
                                                                    classes[4] = 0; 
                                                                    classes[5] = 0; 
                                                                } else {
                                                                    if (features[12] <= 0.228070288897) {
                                                                        if (features[6] <= 0.0890641845763) {
                                                                        } else {
                                                                        }
                                                                    } else {
                                                                        if (features[2] <= 0.15850905329) {
                                                                            classes[0] = 0; 
                                                                            classes[1] = 56; 
                                                                            classes[2] = 0; 
                                                                            classes[3] = 0; 
                                                                            classes[4] = 0; 
                                                                            classes[5] = 0; 
                                                                        } else {
                                                                            if (features[16] <= 0.178904779255) {
                                                                            } else {
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    } else {
                                                        classes[0] = 0; 
                                                        classes[1] = 25; 
                                                        classes[2] = 0; 
                                                        classes[3] = 0; 
                                                        classes[4] = 0; 
                                                        classes[5] = 0; 
                                                    }
                                                } else {
                                                    if (features[7] <= 0.216606721282) {
                                                    } else {
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if (features[1] <= 1.71171486378) {
                                if (features[9] <= 0.311801940203) {
                                    if (features[1] <= 1.43170320988) {
                                        if (features[9] <= 0.288243487477) {
                                        } else {
                                        }
                                    } else {
                                        if (features[9] <= 0.244718007743) {
                                            if (features[12] <= 0.834730923176) {
                                                if (features[21] <= 1.06854301691) {
                                                } else {
                                                }
                                            } else {
                                                classes[0] = 0; 
                                                classes[1] = 0; 
                                                classes[2] = 0; 
                                                classes[3] = 26; 
                                                classes[4] = 0; 
                                                classes[5] = 0; 
                                            }
                                        } else {
                                            if (features[13] <= 0.15361571312) {
                                                if (features[5] <= 0.187254883349) {
                                                } else {
                                                }
                                            } else {
                                                if (features[16] <= 0.180489949882) {
                                                    if (features[14] <= 0.0798727460206) {
                                                    } else {
                                                    }
                                                } else {
                                                    if (features[2] <= 0.164234004915) {
                                                        classes[0] = 0; 
                                                        classes[1] = 71; 
                                                        classes[2] = 0; 
                                                        classes[3] = 0; 
                                                        classes[4] = 0; 
                                                        classes[5] = 0; 
                                                    } else {
                                                        classes[0] = 0; 
                                                        classes[1] = 0; 
                                                        classes[2] = 0; 
                                                        classes[3] = 23; 
                                                        classes[4] = 0; 
                                                        classes[5] = 0; 
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    if (features[7] <= 0.33726605773) {
                                    } else {
                                    }
                                }
                            } else {
                                if (features[12] <= 0.0782368518412) {
                                    if (features[15] <= 0.162483923137) {
                                        classes[0] = 0; 
                                        classes[1] = 0; 
                                        classes[2] = 0; 
                                        classes[3] = 20; 
                                        classes[4] = 0; 
                                        classes[5] = 0; 
                                    } else {
                                        classes[0] = 0; 
                                        classes[1] = 76; 
                                        classes[2] = 0; 
                                        classes[3] = 0; 
                                        classes[4] = 0; 
                                        classes[5] = 0; 
                                    }
                                } else {
                                    if (features[9] <= 0.244155786932) {
                                    } else {
                                    }
                                }
                            }
                        }
                    } else {
                        if (features[9] <= 0.158163696527) {
                            if (features[1] <= 2.65273761749) {
                                if (features[15] <= 0.279904544353) {
                                    if (features[17] <= 0.283930808306) {
                                        if (features[1] <= 2.39365959167) {
                                            if (features[20] <= 0.596402496099) {
                                                if (features[16] <= 0.248216837645) {
                                                } else {
                                                }
                                            } else {
                                                if (features[17] <= 0.280111432076) {
                                                    if (features[23] <= 2.23361599445) {
                                                    } else {
                                                    }
                                                } else {
                                                    classes[0] = 0; 
                                                    classes[1] = 0; 
                                                    classes[2] = 0; 
                                                    classes[3] = 20; 
                                                    classes[4] = 0; 
                                                    classes[5] = 0; 
                                                }
                                            }
                                        } else {
                                            if (features[14] <= 0.281118929386) {
                                                if (features[7] <= 0.138006202877) {
                                                    classes[0] = 0; 
                                                    classes[1] = 40; 
                                                    classes[2] = 0; 
                                                    classes[3] = 0; 
                                                    classes[4] = 0; 
                                                    classes[5] = 0; 
                                                } else {
                                                    if (features[13] <= 0.214926719666) {
                                                    } else {
                                                    }
                                                }
                                            } else {
                                                classes[0] = 0; 
                                                classes[1] = 0; 
                                                classes[2] = 0; 
                                                classes[3] = 126; 
                                                classes[4] = 0; 
                                                classes[5] = 0; 
                                            }
                                        }
                                    } else {
                                        if (features[4] <= 0.115998253226) {
                                            if (features[2] <= 0.105398438871) {
                                                classes[0] = 0; 
                                                classes[1] = 48; 
                                                classes[2] = 0; 
                                                classes[3] = 0; 
                                                classes[4] = 0; 
                                                classes[5] = 0; 
                                            } else {
                                                if (features[7] <= 0.121994003654) {
                                                } else {
                                                }
                                            }
                                        } else {
                                            if (features[6] <= 0.150229215622) {
                                                classes[0] = 0; 
                                                classes[1] = 208; 
                                                classes[2] = 0; 
                                                classes[3] = 0; 
                                                classes[4] = 0; 
                                                classes[5] = 0; 
                                            } else {
                                                if (features[9] <= 0.134258657694) {
                                                    if (features[21] <= 1.03125500679) {
                                                    } else {
                                                    }
                                                } else {
                                                    classes[0] = 0; 
                                                    classes[1] = 47; 
                                                    classes[2] = 0; 
                                                    classes[3] = 0; 
                                                    classes[4] = 0; 
                                                    classes[5] = 0; 
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    if (features[6] <= 0.178615197539) {
                                        if (features[4] <= 0.120931837708) {
                                            if (features[5] <= 0.140483282506) {
                                            } else {
                                            }
                                        } else {
                                            if (features[7] <= 0.0969174951315) {
                                                if (features[15] <= 0.459826827049) {
                                                } else {
                                                }
                                            } else {
                                                if (features[16] <= 0.390487581491) {
                                                    if (features[11] <= 0.276842027903) {
                                                    } else {
                                                    }
                                                } else {
                                                    classes[0] = 0; 
                                                    classes[1] = 0; 
                                                    classes[2] = 0; 
                                                    classes[3] = 18; 
                                                    classes[4] = 0; 
                                                    classes[5] = 0; 
                                                }
                                            }
                                        }
                                    } else {
                                        classes[0] = 0; 
                                        classes[1] = 0; 
                                        classes[2] = 0; 
                                        classes[3] = 18; 
                                        classes[4] = 0; 
                                        classes[5] = 0; 
                                    }
                                }
                            } else {
                                if (features[7] <= 0.108842216432) {
                                } else {
                                }
                            }
                        } else {
                            if (features[1] <= 2.50387907028) {
                                if (features[17] <= 0.198492608964) {
                                    if (features[5] <= 0.0941934287548) {
                                    } else {
                                    }
                                } else {
                                    if (features[7] <= 0.17014978826) {
                                        if (features[24] <= 0.135416664183) {
                                            if (features[10] <= 0.277041301131) {
                                            } else {
                                            }
                                        } else {
                                            if (features[17] <= 0.236936427653) {
                                                if (features[11] <= 0.204520948231) {
                                                    if (features[14] <= 0.290284246206) {
                                                    } else {
                                                    }
                                                } else {
                                                    if (features[1] <= 2.32166004181) {
                                                    } else {
                                                    }
                                                }
                                            } else {
                                                classes[0] = 0; 
                                                classes[1] = 0; 
                                                classes[2] = 0; 
                                                classes[3] = 33; 
                                                classes[4] = 0; 
                                                classes[5] = 0; 
                                            }
                                        }
                                    } else {
                                        if (features[22] <= 1.45664596558) {
                                        } else {
                                        }
                                    }
                                }
                            } else {
                                classes[0] = 0; 
                                classes[1] = 0; 
                                classes[2] = 0; 
                                classes[3] = 435; 
                                classes[4] = 0; 
                                classes[5] = 0; 
                            }
                        }
                    }
                }
            }
        } else {
            if (features[0] <= 0.546511635184) {
                if (features[1] <= 1.13492417336) {
                    if (features[9] <= 0.208865173161) {
                        if (features[17] <= 1.16648578644) {
                            if (features[9] <= 0.160104125738) {
                                if (features[1] <= 0.410771414638) {
                                    if (features[17] <= 0.967116385698) {
                                        if (features[16] <= 0.826751947403) {
                                        } else {
                                        }
                                    } else {
                                        classes[0] = 67; 
                                        classes[1] = 0; 
                                        classes[2] = 0; 
                                        classes[3] = 0; 
                                        classes[4] = 0; 
                                        classes[5] = 0; 
                                    }
                                } else {
                                    if (features[9] <= 0.11741553992) {
                                        if (features[14] <= 0.142547562718) {
                                        } else {
                                        }
                                    } else {
                                        if (features[1] <= 0.945998072624) {
                                            classes[0] = 0; 
                                            classes[1] = 281; 
                                            classes[2] = 0; 
                                            classes[3] = 0; 
                                            classes[4] = 0; 
                                            classes[5] = 0; 
                                        } else {
                                            if (features[10] <= 0.332216784358) {
                                                if (features[17] <= 0.77625182271) {
                                                } else {
                                                }
                                            } else {
                                                if (features[1] <= 1.04284316301) {
                                                    classes[0] = 0; 
                                                    classes[1] = 0; 
                                                    classes[2] = 0; 
                                                    classes[3] = 94; 
                                                    classes[4] = 0; 
                                                    classes[5] = 0; 
                                                } else {
                                                    if (features[14] <= 0.440613433719) {
                                                        classes[0] = 0; 
                                                        classes[1] = 0; 
                                                        classes[2] = 0; 
                                                        classes[3] = 44; 
                                                        classes[4] = 0; 
                                                        classes[5] = 0; 
                                                    } else {
                                                        if (features[8] <= 0.129372671247) {
                                                        } else {
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (features[1] <= 0.874171316624) {
                                    if (features[16] <= 0.919094383717) {
                                        if (features[6] <= 0.330953508615) {
                                        } else {
                                        }
                                    } else {
                                        classes[0] = 0; 
                                        classes[1] = 0; 
                                        classes[2] = 0; 
                                        classes[3] = 27; 
                                        classes[4] = 0; 
                                        classes[5] = 0; 
                                    }
                                } else {
                                    if (features[5] <= 0.181182317436) {
                                        if (features[14] <= 0.425909236073) {
                                        } else {
                                        }
                                    } else {
                                        if (features[2] <= 0.216907314956) {
                                            if (features[12] <= 0.496366217732) {
                                            } else {
                                            }
                                        } else {
                                            if (features[16] <= 0.346304535866) {
                                            } else {
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            if (features[1] <= 1.00683516264) {
                                if (features[6] <= 0.219518654048) {
                                } else {
                                }
                            } else {
                                if (features[8] <= 0.0760814845562) {
                                } else {
                                }
                            }
                        }
                    } else {
                        if (features[4] <= 0.515058577061) {
                        } else {
                        }
                    }
                } else {
                    if (features[9] <= 0.112458765507) {
                        if (features[1] <= 1.41247403622) {
                            if (features[7] <= 0.338174805045) {
                                if (features[17] <= 1.46415734291) {
                                } else {
                                }
                            } else {
                                classes[0] = 0; 
                                classes[1] = 0; 
                                classes[2] = 0; 
                                classes[3] = 26; 
                                classes[4] = 0; 
                                classes[5] = 0; 
                            }
                        } else {
                            if (features[11] <= 0.279457598925) {
                                if (features[7] <= 0.210903733969) {
                                    if (features[9] <= 0.0960171855986) {
                                        classes[0] = 0; 
                                        classes[1] = 99; 
                                        classes[2] = 0; 
                                        classes[3] = 0; 
                                        classes[4] = 0; 
                                        classes[5] = 0; 
                                    } else {
                                        if (features[1] <= 1.56660908461) {
                                            classes[0] = 0; 
                                            classes[1] = 41; 
                                            classes[2] = 0; 
                                            classes[3] = 0; 
                                            classes[4] = 0; 
                                            classes[5] = 0; 
                                        } else {
                                            if (features[20] <= 0.61672398448) {
                                            } else {
                                            }
                                        }
                                    }
                                } else {
                                    classes[0] = 0; 
                                    classes[1] = 0; 
                                    classes[2] = 0; 
                                    classes[3] = 76; 
                                    classes[4] = 0; 
                                    classes[5] = 0; 
                                }
                            } else {
                                if (features[1] <= 1.69563025236) {
                                    if (features[17] <= 0.955093026161) {
                                        if (features[6] <= 0.107356272638) {
                                        } else {
                                        }
                                    } else {
                                        if (features[11] <= 0.314769104123) {
                                        } else {
                                        }
                                    }
                                } else {
                                    classes[0] = 0; 
                                    classes[1] = 0; 
                                    classes[2] = 0; 
                                    classes[3] = 378; 
                                    classes[4] = 0; 
                                    classes[5] = 0; 
                                }
                            }
                        }
                    } else {
                        if (features[1] <= 3.48099434376) {
                            if (features[9] <= 0.432853162289) {
                                if (features[1] <= 1.30980074406) {
                                    if (features[9] <= 0.131073743105) {
                                        if (features[24] <= 0.260416671634) {
                                            if (features[23] <= 2.11520946026) {
                                            } else {
                                            }
                                        } else {
                                            if (features[11] <= 0.410582050681) {
                                            } else {
                                            }
                                        }
                                    } else {
                                        if (features[9] <= 0.179759576917) {
                                            if (features[1] <= 1.18520152569) {
                                                if (features[1] <= 1.16197526455) {
                                                    if (features[19] <= 0.348509997129) {
                                                        if (features[10] <= 0.326111093163) {
                                                        } else {
                                                        }
                                                    } else {
                                                        if (features[2] <= 0.119234573096) {
                                                        } else {
                                                        }
                                                    }
                                                } else {
                                                    classes[0] = 0; 
                                                    classes[1] = 69; 
                                                    classes[2] = 0; 
                                                    classes[3] = 0; 
                                                    classes[4] = 0; 
                                                    classes[5] = 0; 
                                                }
                                            } else {
                                                if (features[0] <= 0.354651167989) {
                                                } else {
                                                }
                                            }
                                        } else {
                                            if (features[6] <= 0.426354885101) {
                                            } else {
                                            }
                                        }
                                    }
                                } else {
                                    if (features[1] <= 3.18523132801) {
                                    } else {
                                    }
                                }
                            } else {
                                if (features[1] <= 2.74531757832) {
                                    if (features[1] <= 2.36864626408) {
                                        classes[0] = 0; 
                                        classes[1] = 0; 
                                        classes[2] = 0; 
                                        classes[3] = 1263; 
                                        classes[4] = 0; 
                                        classes[5] = 0; 
                                    } else {
                                        if (features[2] <= 0.412681356072) {
                                            if (features[9] <= 0.552490055561) {
                                                if (features[15] <= 0.278718963265) {
                                                } else {
                                                }
                                            } else {
                                                if (features[15] <= 0.291034057736) {
                                                } else {
                                                }
                                            }
                                        } else {
                                            if (features[9] <= 0.472583010793) {
                                                classes[0] = 0; 
                                                classes[1] = 0; 
                                                classes[2] = 0; 
                                                classes[3] = 21; 
                                                classes[4] = 0; 
                                                classes[5] = 0; 
                                            } else {
                                                if (features[12] <= 0.221426531672) {
                                                } else {
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    if (features[1] <= 3.23787462711) {
                                        if (features[2] <= 0.118958961219) {
                                            classes[0] = 0; 
                                            classes[1] = 0; 
                                            classes[2] = 0; 
                                            classes[3] = 0; 
                                            classes[4] = 0; 
                                            classes[5] = 58; 
                                        } else {
                                            if (features[13] <= 0.200103223324) {
                                                if (features[17] <= 0.172936186194) {
                                                } else {
                                                }
                                            } else {
                                                if (features[19] <= 0.363889500499) {
                                                    if (features[4] <= 0.43985208869) {
                                                    } else {
                                                    }
                                                } else {
                                                    if (features[17] <= 0.217225916684) {
                                                        if (features[22] <= 1.5674110055) {
                                                        } else {
                                                        }
                                                    } else {
                                                        classes[0] = 0; 
                                                        classes[1] = 0; 
                                                        classes[2] = 0; 
                                                        classes[3] = 23; 
                                                        classes[4] = 0; 
                                                        classes[5] = 0; 
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        if (features[10] <= 0.121898446232) {
                                        } else {
                                        }
                                    }
                                }
                            }
                        } else {
                            if (features[9] <= 0.315623745322) {
                                if (features[7] <= 0.203921988606) {
                                } else {
                                }
                            } else {
                                if (features[16] <= 0.372857987881) {
                                } else {
                                }
                            }
                        }
                    }
                }
            } else {
                if (features[1] <= 1.25918215513) {
                    if (features[9] <= 0.102841425687) {
                        classes[0] = 54; 
                        classes[1] = 0; 
                        classes[2] = 0; 
                        classes[3] = 0; 
                        classes[4] = 0; 
                        classes[5] = 0; 
                    } else {
                        if (features[1] <= 0.913966357708) {
                            classes[0] = 0; 
                            classes[1] = 0; 
                            classes[2] = 0; 
                            classes[3] = 173; 
                            classes[4] = 0; 
                            classes[5] = 0; 
                        } else {
                            if (features[16] <= 0.531561821699) {
                                classes[0] = 0; 
                                classes[1] = 0; 
                                classes[2] = 0; 
                                classes[3] = 0; 
                                classes[4] = 0; 
                                classes[5] = 54; 
                            } else {
                                if (features[14] <= 0.662344396114) {
                                    if (features[5] <= 0.483189448714) {
                                    } else {
                                    }
                                } else {
                                    if (features[6] <= 0.290051311255) {
                                        if (features[18] <= 0.151798002422) {
                                        } else {
                                        }
                                    } else {
                                        classes[0] = 0; 
                                        classes[1] = 0; 
                                        classes[2] = 0; 
                                        classes[3] = 0; 
                                        classes[4] = 0; 
                                        classes[5] = 70; 
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (features[9] <= 0.243205785751) {
                        if (features[1] <= 1.81640958786) {
                            if (features[11] <= 0.224234394729) {
                                if (features[20] <= 0.511913493276) {
                                } else {
                                }
                            } else {
                                classes[0] = 0; 
                                classes[1] = 0; 
                                classes[2] = 0; 
                                classes[3] = 67; 
                                classes[4] = 0; 
                                classes[5] = 0; 
                            }
                        } else {
                            if (features[19] <= 0.401470497251) {
                            } else {
                            }
                        }
                    } else {
                        if (features[13] <= 0.703328877687) {
                        } else {
                        }
                    }
                }
            }
        }

        console.log(classes);
        if (classes[0] === undefined) {
            return (classes.length - classes.length % 2) / 2;
        } 
        else {
            return findMax(classes);
        }
    };

};

export default DecisionTreeClassifier;