if (mr.scriptLoaded && mr.portalSriptLoaded){var regionalSelector = new __portal.PortalRegion({
								effRegion: effRegion,
								region: region,
								auto: !mr.s_cookie.getGlobal('geo'),
								notification: false,
								cityName: 'Канберра',
								startRegion: region.level2

							}
						);

						if (showRegionSelector || mr.hash === 'region'){
							regionalSelector.show();
						}
						mr.bind(document.getElementById('regional-info-link-0'), 'click', function(){
							regionalSelector.show();
						});if (!newsTabs.tabsSwitched && !window.__ie7 && !window.__ie6 && !(!!~navigator.userAgent.indexOf('Opera') && (!~navigator.userAgent.indexOf('Opera/9.8') || !!~navigator.userAgent.indexOf('Version/10')))){

						var regionConfirm = new RegionConfirm({
							show:0,
							cityName: 'Канберра',
							effRegion: effRegion,
							region: region
						});

						regionConfirm.show();regionalSelector.on('beforeShow', function(){
								regionConfirm.hide();
							});}
				}