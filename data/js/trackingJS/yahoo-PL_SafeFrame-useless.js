/*
*
*	Script: PL_SafeFrame.js
*	Version: 1.1.5
*
*	Version: 1.1.1 : 2013-08-20 to add visibility support
*	Version: 1.1.2 : 2013-10-15 to check gEbBC exists before attempting to use it
*   Version: 1.1.3 : 2014-02-10 added support for StdBanners (it wasn't expected the safeframe script would be used on a StdBanner)
*   Version: 1.1.4 : 2014-03-14 modified to work with new client visibility changes
*   Version: 1.1.5 : 2014-03-23 further visibility changes to work in approved new client visibility interfacing method
*
*/

//if we're a SafeFrame (regardless of old/new client) create the window.SafeFrameObj;
try{
	if(((window.$sf && $sf.ext) || (window.Y && Y.SandBox && Y.SandBox.vendor)) && !(window["$WLXRmAd"])) {
		ebO.ifrm=0;
		window.SafeFrameObj = {
			uid: ebAdID + "_" + ebRand,
			iabSF: window.$sf && $sf.ext,
			yahooSF: window.Y && Y.SandBox && Y.SandBox.vendor,
			SFapi: null,
			oldClient: !ebO.extensionHooks,
			newClient: !!ebO.extensionHooks,
			handleException: function(s) {
				//console.log("SafeFrameError: "+s);
			},
			properties: undefined,
			push: false,
			awaitingExpandComplete: false,
			panels: { banner: { w:ebO.w, h:ebO.h } },
			ignoreExpandWithoutPanelName: !ebO.extensionHooks, //if old client, true, if new client, false
			currentPanel: undefined,
			adObj: null,
			newClientRecalc: null,
			newClientCC: null,
			lastPercentage: -1,
			percentageSubscribers: [],
			geom: null,
			setPercentageSubscriber: function(obj,func) {
				for(var i=0;i<SafeFrameObj.percentageSubscribers.length;i++) {
					if(obj == SafeFrameObj.percentageSubscribers[i].obj && func == SafeFrameObj.percentageSubscribers[i].func)
						return;													//already in the list, don't re-add it
				}
				SafeFrameObj.percentageSubscribers.push({obj:obj,func:func});
			},
			checkPercentage: function() {
				var currentPercentage = SafeFrameObj.SFapi.inViewPercentage();
				SafeFrameObj.geom = SafeFrameObj.SFapi.geom();
				if(currentPercentage != SafeFrameObj.lastPercentage) {			//if percentage is different, fake a scroll event
					for(var i=0;i<SafeFrameObj.percentageSubscribers.length;i++) {
						if(SafeFrameObj.percentageSubscribers[i].obj) {
							SafeFrameObj.percentageSubscribers[i].obj[SafeFrameObj.percentageSubscribers[i].func]();
						}
						else {
							SafeFrameObj.percentageSubscribers.splice(i,1);		//remove a dead subscriber
						}
					}
				}
				SafeFrameObj.lastPercentage = currentPercentage;
			},
			adjustPanelPos: function() {
				if(SafeFrameObj.oldClient){
					SafeFrameObj.adObj.ad.panels[SafeFrameObj.currentPanel].panelDiv.style.left="0px";
					SafeFrameObj.adObj.ad.panels[SafeFrameObj.currentPanel].panelDiv.style.top="0px";
				}
			},
			fixSafariRedraw:function() {
				if(SafeFrameObj.newClientCC && SafeFrameObj.newClientCC.fixSafariRedrawIssue) {
					var s = "SafeFrameObj.newClientCC.fixSafariRedrawIssue.apply(SafeFrameObj.newClientCC,[])";
					setTimeout(s,10);setTimeout(s,100);setTimeout(s,250);setTimeout(s,1000);
				}
			},
			setPanel: function(param) {
				if(SafeFrameObj.oldClient) {	//if old client, we get param name
					SafeFrameObj.currentPanel = param.toLowerCase();
					SafeFrameObj.push = SafeFrameObj.adObj.ad.panels[SafeFrameObj.currentPanel].fPushDown;
				}
				else if(param.dispatcher._adConfig.uid == SafeFrameObj.uid) {	//if new client, we get event with panel name in it
					SafeFrameObj.currentPanel = param.eventData.props.panel.name;
					SafeFrameObj.push = param.dispatcher.isPushDownPanel;

					//overwrite this function because it gets confused about SafeFrames
					if(!param.dispatcher.orig_calculatePositionHelper) {
						param.dispatcher.orig_calculatePositionHelper = param.dispatcher.calculatePositionHelper;
						SafeFrameObj.newClientRecalc = param.dispatcher.recalculatePanelPosition;
						SafeFrameObj.newClientCC = param.dispatcher;

						param.dispatcher.calculatePositionHelper = function(){
							if(this._panelExpanded)
								return {Left:0,Top:0};
							else
								return this.orig_calculatePositionHelper.apply(this,arguments);
						}
					}
				}
			},
			init: function(obj) {	//this is called by the $wlx initialization code in the client/delivery
				SafeFrameObj.properties = obj;	//save the properties that the client's _wlxInit function calculated

				//allow for the possibility of the ad format adding some customization to the init process
				if(SafeFrameObj.initOverride)
					SafeFrameObj.initOverride(SafeFrameObj);

				try{SafeFrameObj.SFapi.register(SafeFrameObj.properties.width, SafeFrameObj.properties.height, SafeFrameObj.notify);}
				catch(e){SafeFrameObj.handleException(e.message);}
			},
			expand: function(panelName, params) {	//Default expand function for non-custom-formats
				var args = arguments.length;
				var panel = SafeFrameObj.currentPanel;	//assume we're going to act on the most recently set currentPanel
				if(args>0)
					panel = panelName.toLowerCase();	//override that assumption if we received a panelName as a parameter
				else if(SafeFrameObj.ignoreExpandWithoutPanelName)	//if we're in ignore expand calls without params, give up now
					return;
				if(!panel || !SafeFrameObj.adObj)return;	//if we didn't get a panel name by now, there's no hope

				SafeFrameObj.panels[panel].expanded = true;
				SafeFrameObj.adjustPanelPos();
				var pan = SafeFrameObj.panels[panel];
				var ban = SafeFrameObj.panels.banner;
				var xDiff = pan.w - ban.w;
				var yDiff = pan.h - ban.h;
				if(xDiff == 0 && yDiff == 0)
					return; //nothing to do if panel same size as banner (e.g. single expandable)

				var obj = {push:SafeFrameObj.push};
				if(args>1)
					obj = params;
				else {
					if(xDiff > 0) {						//if we've got width expansion
						if(pan.x < 0) {					//if original panel offset was negative, we're going left
							obj.l = Math.abs(pan.x);	//by that amount of pixels (made positive for obj.l)
							if(xDiff > obj.l)			//if obj.l doesn't represent ALL of the width expansion, we have more...
								obj.r = xDiff - obj.l;	//remainder of expansion must go right if not expanding the full width left
						}
						else obj.r = pan.x + xDiff;
					}
					if(yDiff > 0) {						//if we've got height expansion
						if(pan.y < 0) {					//if original panel offset was negative, we're going up
							obj.t = Math.abs(pan.y);	//by that amount of pixels (made positive for obj.t)
							if(yDiff > obj.t)			//if obj.t doesn't represent ALL of the height expansion, we have more...
								obj.b = yDiff - obj.t;	//remainder of expansion must go down if not expanding the full width up
						}
						else obj.b = pan.y + yDiff;
					}
				}
				var dX = xDiff > 0 ? (pan.x < 0 ? -xDiff:xDiff) : 0;
				var dY = yDiff > 0 ? (pan.y < 0 ? -yDiff:yDiff) : 0;
				SafeFrameObj.awaitingExpandComplete = true;
				if(SafeFrameObj.iabSF) {
					try{SafeFrameObj.SFapi.expand(obj);}catch(e){SafeFrameObj.handleException(e.message);}
				}
				else if(SafeFrameObj.yahooSF) {	//if we have to use yahoo API, use the two param system (left OR right, up OR down), max 2 expand directions
					try{SafeFrameObj.SFapi.expand(dX, dY, SafeFrameObj.push);}catch(e){SafeFrameObj.handleException(e.message);}
				}
				SafeFrameObj.fixSafariRedraw();
			},
			collapse: function(panelName) {
				SafeFrameObj.awaitingExpandComplete = false;
				if(SafeFrameObj.currentPanel)
					SafeFrameObj.panels[SafeFrameObj.currentPanel].expanded = false;

				try{SafeFrameObj.SFapi.collapse();}catch(e){SafeFrameObj.handleException(e.message);}
				SafeFrameObj.fixSafariRedraw();
			},
			notify: function(message, info) {									//event driven asynchronous response system to earlier commands
				var p = SafeFrameObj.currentPanel;

				if(message=="expanded" || message=="failed") {
					SafeFrameObj.awaitingExpandComplete = false;
					if(message=="expanded")
						SafeFrameObj.adjustPanelPos();
				}

				//handle a failed expand, or a request (by the host) to collapse
				if(p && SafeFrameObj.panels[p].expanded) {						//if we're expanded
					if(message == "collapsed" || message == "failed") {			//we failed to expand or "they" collapsed us
						SafeFrameObj.panels[p].expanded = false;				//this is important to do right here to prevent infinte collapse loop
						var func = SafeFrameObj.newClient ? "_hidePanel" : "hidePanel";
						SafeFrameObj.adObj && !SafeFrameObj.adObj.isSEAd() && SafeFrameObj.adObj[func](p,true);
					}
				}

				//handle a successful expand or collapse for old client
				//if the message is collapsed, all ad types and browsers want the resize
				//if the message is expanded, all browsers except opera want the resize (as long as it's not a singleexpbanner ad)
				if(SafeFrameObj.oldClient && message=="collapsed" && SafeFrameObj.adObj.isSEAd()) {
					SafeFrameObj.adObj.ad.panels[SafeFrameObj.currentPanel].doOnResize(0,0);
				}
				if(SafeFrameObj.newClient && message=="collapsed" && SafeFrameObj.newClientRecalc) {
					SafeFrameObj.newClientRecalc.apply(SafeFrameObj.newClientCC,[]);
				}
			}
		}
		SafeFrameObj.SFapi = SafeFrameObj.iabSF || SafeFrameObj.yahooSF;

		//switch on the hook to SafeFrameObj by defining $WLXRmAd which the old and new client will see and think it's MSN WLX interface
		window["$WLXRmAd"] = SafeFrameObj;	//create alias to ensure existing old/new client wlx functions are routed into this object

		if(SafeFrameObj.oldClient) {
			//========================================================
			//Standard old client "merge" ebCCustomEventHandlers code
			//========================================================

			SafeFrameObj.percentageInterval = setInterval(SafeFrameObj.checkPercentage,1000);	//check the percentage once per second

			var mmRand = Math.floor(Math.random() * 99999999999);
			var mmAdId = ebO.ai;
			window["mmOriginalEventHandlers_" + mmRand] = window.ebCCustomEventHandlers ? new ebCCustomEventHandlers() : undefined;
			window.mmCustomEventHandlers = function() {
				if(SafeFrameObj){
					//if IE9+ and in a SafeFrame, we need to re-subscribe to onload event handler using our own subscribeToEvent
					//function which has the if attachEvent else addEventListener structure, because the default delivery method of first
					//checking if addEventListener causes addEventListener to be used on IE9+ (since they decided to start supporting it
					//in IE9). The problem is that Yahoo have overridden the handling of it somehow and their addEventListener code is
					//somehow causing us to not get the load event. Simply using attachEvent instead works fine.
					if(window.gEbBC && gEbBC.isIE() && gEbBC.getVersion()>=9) {
						subscribeToEvent(window, "load", ebOnloadHandler);
					}
				}
				var oEH = window["mmOriginalEventHandlers_" + mmRand];
				this.onClientScriptsLoaded=		function(objName) {
					if(SafeFrameObj){
						subscribeToEvent(window, "unload", SafeFrameObj.collapse);
						subscribeToEvent(window, "beforeunload", SafeFrameObj.collapse);

						window.orig_ebOnmouseOutOfAd = window.ebOnmouseOutOfAd;
						window.ebOnmouseOutOfAd = function() { if(SafeFrameObj.awaitingExpandComplete)return; else return window.orig_ebOnmouseOutOfAd.apply(this,arguments); }

						//Override the ad constructor because we want to tweak the objects it creates
						var orig_ad_constructor = window["ebAd_"+mmAdId];
						window["ebAd_"+mmAdId] = function() {
							var ebAd = new orig_ad_constructor();	//create an ad object using the old constructor
							for(var pan in ebAd.panels) {			//and then tweak the values for SafeFrames
								if(ebAd.panels.hasOwnProperty(pan)) {
									var panlo = pan.toLowerCase();
									SafeFrameObj.panels[panlo] = { expanded:false, firstExpand:true,
										x:ebAd.panels[pan].nXPos,  y:ebAd.panels[pan].nYPos,
										w:ebAd.panels[pan].nWidth, h:ebAd.panels[pan].nHeight
									};
								}
							}
							return ebAd;
						}

						//Override the clientArea constructor because we want to tweak the objects it creates
						var orig_clientArea_constructor = window.ebCClientArea;
						window.ebCClientArea = function(obj) {
							var clientArea = new orig_clientArea_constructor(obj);

							//now override the isVisible function
							clientArea.isVisible = function() {
								if(SafeFrameObj.geom){
									this.nHeight = parseInt(SafeFrameObj.geom.win.h);
									this.nWidth = parseInt(SafeFrameObj.geom.win.w);
								}
								var percentage = SafeFrameObj.SFapi.inViewPercentage();
								return percentage;
							}

							return clientArea;
						}
					}
					try{oEH.onClientScriptsLoaded.apply(this,arguments)}catch(e){}
				};
				this.onBeforePanelShow=			function(objName, panelName) {
					if(SafeFrameObj){
						var panlo = panelName.toLowerCase();
						SafeFrameObj.setPanel(panelName);
						if(!SafeFrameObj.adObj.isSEAd()) {	//if not SingleExpBanner, this is an expand panel
							SafeFrameObj.expand(panelName);	//we call $WLXRmAd.expand here even though it was called already by the old client
															//because when the client did it, the panelName wasn't known yet. Now it is, we call
															//expand again. We know the last call to expand won't have "done anything" because it
															//was called without parameters, and that causes an early return. Now we call it with panelName

							var IE = gEbBC.isIE();
							var IE9p = IE && gEbBC.getVersion()>=9;
							var DM9p = document.documentMode>=9;
							var qurk = document.compatMode !== "CSS1Compat";
							var nEx1 = !SafeFrameObj.panels[panlo].firstExpand;
							var weirdIEmatrix = (IE9p && DM9p && nEx1);//IE doesn't need this fix except if it's IE9+ in document mode 9+ and only if it's not the first expand
							var thisFixNeeded = gEbBC.isChrome() || gEbBC.isSafari() || gEbBC.isFF() || weirdIEmatrix;
							SafeFrameObj.panels[panlo].firstExpand = false;

							if(thisFixNeeded){
								var pans = SafeFrameObj.adObj.ad.panels;
								var p = pans[panelName] || pans[panlo];
								if(p){
									p.nXPos = SafeFrameObj.panels[panlo].expanded ? 0 : SafeFrameObj.panels[panlo].nXPos;
									p.nYPos = SafeFrameObj.panels[panlo].expanded ? 0 : SafeFrameObj.panels[panlo].nYPos;
								}
							}
						}
						else {
							//We're a single exp banner. By the time we get here (onbeforepanelshow) we know that wlxExpand has already been called
							//once without parameters. That was the one we needed our SafeFrameObj.expand function to ignore. We also know that the
							//code above has now waved the panelName. So we're safe to allow SafeFrameObj.expand to accept calls to it without any
							//params to be accepted, because that's all we'll get from now on for a single exp banner. There'll be no more calls to
							//onbeforepanelshow (it's only "shown" once). The ebseexpandstartedHandler and ebseretractfinishedHandler will do the
							//calls for us to SafeFrameObj.expand/collapse without params, but as the one and only call to onbeforepanelshow set the
							//one and only panel, we're good.
							SafeFrameObj.ignoreExpandWithoutPanelName = false;
						}
					}
					try{oEH.onBeforePanelShow.apply(this,arguments)}catch(e){}
				};
				this.onBeforeAddRes=			function(objName) {					try{oEH.onBeforeAddRes.apply(			this,arguments)}catch(e){}};
				this.onBeforeAddPanelRes=		function(objName, panelName) {		try{oEH.onBeforeAddPanelRes.apply(		this,arguments)}catch(e){}};
				this.onHandleInteraction=		function(objName,intName,strObjID){	try{oEH.onHandleInteraction.apply(		this,arguments)}catch(e){}};
				this.onBeforeDefaultBannerShow=	function(objName) {
					try{
						if(SafeFrameObj){
							SafeFrameObj.adObj = eval(objName);
							SafeFrameObj.setPercentageSubscriber(SafeFrameObj.adObj,"doOnScroll");
						}
						oEH.onBeforeDefaultBannerShow.apply(this,arguments)
					}
					catch(e){}
				};
				this.onAfterDefaultBannerShow=	function(objName) {					try{oEH.onAfterDefaultBannerShow.apply(	this,arguments)}catch(e){}};
				this.onBeforeRichFlashShow=		function(objName) {					try{oEH.onBeforeRichFlashShow.apply(	this,arguments)}catch(e){}};
				this.onAfterRichFlashShow=		function(objName) {					try{oEH.onAfterRichFlashShow.apply(		this,arguments)}catch(e){}};
				this.onAfterPanelShow=			function(objName, panelName) {		try{oEH.onAfterPanelShow.apply(			this,arguments)}catch(e){}};
				this.onBeforePanelHide=			function(objName, panelName) {
					if(SafeFrameObj){
						SafeFrameObj.collapse();	//TODO: wait for the pushdown's "pullup" operation to complete?
					}
					try{oEH.onBeforePanelHide.apply(		this,arguments)}catch(e){}
				};
				this.onAfterPanelHide=			function(objName, panelName) {		try{oEH.onAfterPanelHide.apply(			this,arguments)}catch(e){}};
				this.onBeforeAdClose=			function(objName) {					try{oEH.onBeforeAdClose.apply(			this,arguments)}catch(e){}};
				this.onAfterAdClose=			function(objName) {					try{oEH.onAfterAdClose.apply(			this,arguments)}catch(e){}};
				this.onBeforeIntroShow=			function(objName) {					try{oEH.onBeforeIntroShow.apply(		this,arguments)}catch(e){}};
				this.onAfterIntroShow=			function(objName) {					try{oEH.onAfterIntroShow.apply(			this,arguments)}catch(e){}};
				this.onBeforeIntroHide=			function(objName) {					try{oEH.onBeforeIntroHide.apply(		this,arguments)}catch(e){}};
				this.onAfterIntroHide=			function(objName) {					try{oEH.onAfterIntroHide.apply(			this,arguments)}catch(e){}};
				this.onBeforeRemShow=			function(objName) {					try{oEH.onBeforeRemShow.apply(			this,arguments)}catch(e){}};
				this.onAfterRemShow=			function(objName) {					try{oEH.onAfterRemShow.apply(			this,arguments)}catch(e){}};
				this.onBeforeRemHide=			function(objName) {					try{oEH.onBeforeRemHide.apply(			this,arguments)}catch(e){}};
				this.onAfterRemHide=			function(objName) {					try{oEH.onAfterRemHide.apply(			this,arguments)}catch(e){}};
				this.onBeforeMiniSiteShow=		function(objName) {					try{oEH.onBeforeMiniSiteShow.apply(		this,arguments)}catch(e){}};
				this.onAfterMiniSiteShow=		function(objName) {					try{oEH.onAfterMiniSiteShow.apply(		this,arguments)}catch(e){}};
				this.onBeforeMiniSiteHide=		function(objName) {					try{oEH.onBeforeMiniSiteHide.apply(		this,arguments)}catch(e){}};
				this.onAfterMiniSiteHide=		function(objName) {					try{oEH.onAfterMiniSiteHide.apply(		this,arguments)}catch(e){}};

				function subscribeToEvent(el,evt,cb){el.attachEvent?el.attachEvent("on"+evt,cb):el.addEventListener(evt,cb,false);}
			}
			window.ebCCustomEventHandlers = window.mmCustomEventHandlers;
		}
		else {	//new client
			(function(){
				var adId = window.ebAdID || 0;  //this is the ad onto which we're piggybacking
				var rnd = ebRand;
				var uid = adId + "_" + rnd;
				ebO.extensionHooks.push(function()
				{
					var tn = ebO.adConfig.templateName;
					var exp = tn=="ExpBanner";
					var sex = tn=="SingleExpBanner"

					if(SafeFrameObj){
						//========================================================
						//During extension hooks (long before the ad is created),
						//record the panel left/top, and then override them to 0.
						//========================================================
						if(exp)
							EBG.Ads.ExpBanner.requiresIframeBust = false;	//ensure ExpBanner knows not to breakout, we must do this as early as extension hooks
						if(sex)
							EBG.Ads.SingleExpBanner.requiresIframeBust = false;	//ensure SingleExpBanner knows not to breakout, we must do this as early as extension hooks
						for(var pan in ebO.adConfig.panels) {
							if(ebO.adConfig.panels.hasOwnProperty(pan)) {
								var p = ebO.adConfig.panels[pan];
								SafeFrameObj.panels[pan.toLowerCase()]={expanded:false,x:sex?p.xPos*-1:p.xPos,y:sex?p.yPos*-1:p.yPos,w:p.width,h:p.height};
							}
						}

						//========================================================
						//Subscribe to EXPAND [panel] so we can record which panel
						//is expanding, and override calculatePositionHelper
						//========================================================
						var createExpandPanelSubscription = new EBG.Events.EventSubscription(EBG.Events.EventNames.EXPAND,SafeFrameObj.setPanel);
						createExpandPanelSubscription.timing = EBG.Events.EventTiming.BEFORE;
						EBG.eventMgr.subscribeToEvent(createExpandPanelSubscription);

						//========================================================
						//Subscribe to CREATE_ADAPTOR so we can override the funcs
						//that look for the current window. In a SafeFrame it's
						//always "window". Doing this stops security exceptions
						//========================================================
						var createExpandPanelSubscription = new EBG.Events.EventSubscription(EBG.Events.EventNames.CREATE_ADAPTOR,function(event){
							//override these functions that were getting confused (or causing security exceptions). In safeframe it's always window
							//we have to do this after the adaptor was created but not any later than that (and we can't do it any earlier)
							EBG.adaptor.getDisplayWin =		function(){return window;}
							EBG.adaptor.getPageWin =		function(){return window;}
							EBG.adaptor._getCurrentWindow =	function(){return window;}
							EBG.adaptor.getHostName =		function(){return window.location.hostname;}
							EBG.adaptor._addEventListener = function(target, eventName, callback) {	//reverse eventlistener/attachevent logic
						        if (eventName == "load") {
						        	if(this.isPageLoaded()) {
							            setTimeout(function () { callback({ type: "load" }) }, 5);
							            return;
							        }
							        else {
								        if (window.attachEvent) {
								            target.attachEvent("on" + eventName, callback);
								        }
								        else if (window.addEventListener) {
								            target.addEventListener(eventName, callback, false);
								        }
							        }
						        }
						        if (window.addEventListener) {
						            target.addEventListener(eventName, callback, false);
						        }
						        else if (window.attachEvent) {
						            target.attachEvent("on" + eventName, callback);
						        }
		        			}
						});
						createExpandPanelSubscription.timing = EBG.Events.EventTiming.AFTER;
						EBG.eventMgr.subscribeToEvent(createExpandPanelSubscription);

						//========================================================
						//Subscribe to SHOW_AD so we can then subscribe to the
						//unload event(s) and call SFapi.collapse to clean up
						//========================================================
						var createAdSubscription = new EBG.Events.EventSubscription(EBG.Events.EventNames.SHOW_AD,
							function(event){
								if(event.dispatcher._adConfig.uid == uid) {
									SafeFrameObj.adObj = event.dispatcher;
									var pageUnloadSubscription = new EBG.Events.EventSubscription(EBG.Events.EventNames.PAGE_UNLOAD, SafeFrameObj.collapse);
									pageUnloadSubscription.timing = EBG.Events.EventTiming.BEFORE;
									EBG.eventMgr.subscribeToEvent(pageUnloadSubscription);

									var pageBeforeUnloadSubscription = new EBG.Events.EventSubscription(EBG.Events.EventNames.PAGE_BEFORE_UNLOAD, SafeFrameObj.collapse);
									pageBeforeUnloadSubscription.timing = EBG.Events.EventTiming.BEFORE;
									EBG.eventMgr.subscribeToEvent(pageBeforeUnloadSubscription);

									if(EBG.adaptor.browser.isMac() && EBG.adaptor.browser.isSafari() && event.dispatcher._adConfig.templateName == "SingleExpBanner") {
										EBG.RichFlashPanelCC.prototype.orig_checkShowHideElements = EBG.RichFlashPanelCC.prototype._checkShowHideElements;
										EBG.RichFlashPanelCC.prototype.fixSafariRedrawIssue = function() {
											var divToFix = EBG.adaptor.getElementById(this.props.panel.id);
											if(divToFix)	// bug fix for mac webkit redraw issue, changing opacity to .99 then back to 1 is barely visible to the user
											{				// but enough to force webkit to redraw the css changes, we must do it via timeout, otherwise redraw does not occur
												divToFix.style.opacity = '.99';
												var restoreOpacity = function () { divToFix.style.opacity = 1; };
												setTimeout(restoreOpacity, 1);
											}
										}
										EBG.RichFlashPanelCC.prototype._checkShowHideElements = function() {
											this.orig_checkShowHideElements();
											this.fixSafariRedrawIssue();
										}
									}

									EBG.declareNamespace("RichModules");
									EBG.RichModules.SafeFrameVisibilityProvider = function (resourceObjId, adConfig) {
									    EBG.callSuperConstructor(EBG.RichModules.SafeFrameVisibilityProvider, this, [resourceObjId, adConfig]);
									    this._initSafeFrameObject();
									    this._intervalHandler = EBG.setInterval(this, this._checkVisibility, [], EBG.RichModules.VisibilityManager.VisibilityPollingTimer);
									};
									EBG.RichModules.SafeFrameVisibilityProvider._isAvailable = function (adConfig) {
									    return !!(window.$sf && $sf.ext);
									};
									EBG.RichModules.SafeFrameVisibilityProvider.prototype = {
									    _safeFrameObj: null,
									    _intervalHandler: null,
									    _lastPercentage: null,
									    _lastViewport: null,
									    _initSafeFrameObject:function() {
									        this._safeFrameObj = window.$sf && $sf.ext || window.Y && Y.SandBox && Y.SandBox.vendor;
									    },
									    _calculateVisibilityPercentage: function () {
									        try {
									            return this._safeFrameObj.inViewPercentage();
									        } catch (e) {
									            return 0;
									        }
									    },
									    _getViewPortRect: function () {
									        var geom = this._safeFrameObj.geom().win;
									        return {w: parseInt(geom.w), h: parseInt(geom.h), w_h: geom.w + "_" + geom.h };
									    },
									    _getViewPortMetrics: function () {
									        var rect = this._getViewPortRect();
									        return { Height: rect.h, Width: rect.w };
									    },
									    _isVisible: function () {
									        return !!this._calculateVisibilityPercentage();
									    },
									    _subscribeToEvents: function () {
									        EBG.callSuperFunction(EBG.RichModules.SafeFrameVisibilityProvider, this, "_subscribeToEvents");
									        EBG.eventMgr.subscribeToEvent(new EBG.Events.EventSubscription(EBG.Events.EventNames.MAX_AD_DURATION, this._maxAdDurationHandler, this));
									    },
									    updateResourceObjId: function (resourceObjId) {
									        EBG.callSuperFunction(EBG.RichModules.SafeFrameVisibilityProvider, this, "updateResourceObjId", [resourceObjId]);
									    },
									    _checkVisibility: function () {
									        if (!this._isStarted)return;
									        var percentage = this._calculateVisibilityPercentage();
									        var viewport = this._getViewPortRect().w_h;
									        if(this._lastPercentage != percentage || this._lastViewport != viewport) {
									            var res = this._calculateVisibilityResult();
									            this._dispatchCheckVisibility(res);
									        }
									        this._lastPercentage = percentage;
									        this._lastViewport = viewport;
									    },
									    _maxAdDurationHandler: function () {
									        if (this._intervalHandler) { EBG.clearInterval(this._intervalHandler); }
									        EBG.log.debug("Caught max duration time event");
									    },
									    start: function () {
									        EBG.callSuperFunction(EBG.RichModules.SafeFrameVisibilityProvider, this, "start");
									    }
									};
									EBG.declareClass(EBG.RichModules.SafeFrameVisibilityProvider, EBG.RichModules.BasicVisibilityProvider);
									EBG.RichModules.VisibilityManager.prototype._availableVisibilityProviders = [EBG.RichModules.SafeFrameVisibilityProvider];	//There can be only one, this one
								}
							}
						);
						createAdSubscription.timing = EBG.Events.EventTiming.BEFORE;
						EBG.eventMgr.subscribeToEvent(createAdSubscription);
					}
				});
			})();
		}
	}
}catch(e){
	SFError = e;
}
