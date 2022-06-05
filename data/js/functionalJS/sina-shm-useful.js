
    jsLoader({
        name: 'shm',
        url: 'http://www.sina.com.cn/js/index/96/0801/shm.js'
    });

                jsLoader({
                    name: 'shm',
                    callback: function(){
                        jsLoader({
                            name: 'StateMgr',
                            url: 'http://www.sina.com.cn/js/96/2014/0317/StateMgr.js',
                            callback:function(){
								var guessCtr=document.getElementById('SI_Guess_Wrap');
                                var mgr = new SHM.xy.stateMgr({
                                    timeSlice: 0, //min
                                    state1Ids:['xy-tabA', 'xy-contA', 'xy-imptabtp-A', 'xy-impcon-A'],
                                    state2Ids:['xy-tabB', 'xy-contB', 'xy-imptabtp-B', 'xy-impcon-B'],
                                    s1Callback:function(){
										//guessCtr.style.height='186px';//8 条
                                        SHM.app.tab.switchByEle(SHM.E('video-tab'));
try{_S_uaTrack("www_update_div", "a");}catch(e){}
                                    },
                                    s2Callback:function(){
										//guessCtr.style.height='186px';//8 条
                                        SHM.app.tab.switchByEle(SHM.E('zhuanlan-tab'));
try{_S_uaTrack("www_update_div", "b");}catch(e){}
                                    }
                                });
                                SHM.evt.addEvent(SHM.E('xy-change'), 'click', function(evt){
                                    mgr.toggleState();
                                    SHM.evt.preventDefault(evt);
                                }, false);
                            }
                        });
                    }
                });
