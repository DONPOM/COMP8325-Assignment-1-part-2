
		require(["apps/App", "apps/PagesApp"]);

		$("body").on("dock:tab", function(e, data){
			$(".dock-pane-tab").addClass("hidden");
			var el = $(".dock-pane-tab."+data.name).removeClass("hidden");

			require(["apps/PagesApp"], function(){
				el.trigger("displayed");
			});
		});

		$(".paneDoneButton").on("click", function() {
			window.parent.jQuery(window.parent.document).trigger('pagesDone');
		});
	