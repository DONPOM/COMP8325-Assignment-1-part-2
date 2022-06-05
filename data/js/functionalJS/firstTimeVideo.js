define([
	'translate!webs.builder.cobrand.webs'
], function (translate) {
	return {
		showPopover: function (videoUrl) {
			var content = '<div id="webs-first-time-video"><iframe width="800" height="450" src="' + videoUrl + '" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe></div>';
			var videoPopover = document.createElement('div');
			videoPopover.setAttribute("id", "firsttimevideo_popover");
			videoPopover.innerHTML = content;
			return new Popover(videoPopover, {
				width: 820,
				height: 470,
				heading: translate("webs.builder.cobrand.webs.welcome.heading"),
				onClose: function() {
					if (bldr.pageController.trees.find("webs-body").model.hasSinglePlaceholder()) {
						require("internal/sitebuilder/builderChrome/chromeController").showToasterHelptip();
					}
				}
			}).show();
		}
	};
});
