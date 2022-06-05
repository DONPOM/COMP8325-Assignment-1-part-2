define("sb3Branding",[
	'jquery',
	'translate!webs.bldr.dock.ads',
	'translate!webs.builder.cobrand.webs',
	'translate!webs.freebar.stickyFreebar',
	'translate!webs.appStore'
],function($, translate){
	return ({
		getFreebarHTML: function(){
			return (
				'<div>' +
					'<span>' + translate('webs.freebar.stickyFreebar.linkTitle') + '</span>' +
					'<a id="remove-freebar" class="w-btn3 red"><span>' + translate('webs.freebar.stickyFreebar.remove') + '</span></a>' +
				'</div>'
			);
		},
		onClickFreebar: function(){
			webs.showPremiumDialog('footer');
		},
		addAds: function(dockController){
			$.ajax({
				url: "/s/sitebuilder/api/installedApps",
				dataType: "json",
				cache: false,
				success: function(data){
					if (data.indexOf("Web Store") === -1) {
						dockController.addAd("webs.bldr.modules.container.commerce", {
							eventName: "webStore",
							text: 'Want to build an Online Store?',
							buttonText: 'Install our Web Store App',
							icon: 'webStoreApp',
							buttonColor:'green',
							target: "/s/app/install?section=app&action=install&app=webstore&next=hidden&title=" + translate('webs.appStore.webstore')
						});
					}
				}
			});
		},

		showFirstTimeExperience: function(){
			//check if we show video or the walkthrough tour
			if (webs.site.firstTimeShowVideo) {
				var videoUrl = translate("webs.builder.cobrand.webs.firstTimeVideo.url");

				if (videoUrl && videoUrl !== 'null' &&
					// only show first time video to english-speaking Webs users:
					webs && /^en/.test(webs.locale)
				) {
					require(['cobrand/webs/firstTimeVideo'],function(firstTimeVideo){
						firstTimeVideo.showPopover(videoUrl);
					});
				}
				//return true.  we don't want to show the walkthrough tour
				return true;
			} else {
				//return false to show the walkthrough tour
				return false;
			}
		},

		showStoreUpsell: function(){}
	});
});