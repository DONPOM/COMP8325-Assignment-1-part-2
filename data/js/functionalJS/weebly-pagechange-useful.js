
function changePage() {
	try {
		var page = "main";
		//console.log('changePage: ' + page);
		var m = page.match(/^themesMenu(\/(.+))?$/);
		if (m) {
			// outsource histroy processing to theme browser...
			parent.litePageChange = true;
			parent.Pages.go('themesMenu', "undefined", "undefined", "undefined", "undefined", "undefined");
			parent.litePageChange = false;
			parent.processDesignHistory && parent.processDesignHistory(m[2]);
		}else{
			if (parent.closeThemePreview) {
				// b/c theme browser previewing isn't using the Pages.go system, it is sometimes left open by accident
				parent.closeThemePreview();
			}
			parent.Pages.go(page, "undefined", "undefined", "undefined", "undefined", "undefined");
		}
	} catch (err) {
		//alert(err);
	}
}
