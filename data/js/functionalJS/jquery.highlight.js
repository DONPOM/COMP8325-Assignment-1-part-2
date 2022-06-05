(function (factory) {
    if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module.
        define(['jquery'], factory);
    } else {
        // Browser globals
        factory(jQuery);
    }
}(function ($) {
	// Highlight dom elements by putting a shadow over all other elements.
	//
	// Places a div.shadow inside each positioned parent, and sets the z-indices correctly
	//  assumes div.shadow has top/left/right/bottom:0, position:absolute, and a bg.

	// lowest z-index used by the plugin, should be above all z-indices in the normal page
	var FIRST_Z = 1000000,

		// class names used to mark things we change
		shadowClassName = "jq-highlight-shadow",
		ghostClassName = "jq-highlight-ghost",
		movedClassName = 'jq-highlight-moved',
		placedAboveClassName = 'jq-highlight-placed-above';

	// get a jQuery set of all positioned parents of the first item in the current jQuery set
	//   returns them in top-down order.
	$.fn.positionedParents = function(){
		var node = this,
			positionedParents = $([]),
			prevNode;

		do {
			prevNode = node;
			node = node.offsetParent();
			positionedParents = positionedParents.add(node);
		} while (node[0] != prevNode[0])

		return positionedParents;
	};


	$.fn.highlight = function(opts){
		var parents = this.positionedParents(),
			z = FIRST_Z;
		parents.each(function(i, container){
			var $c    = $(container),
				oldZ  = $c.css('zIndex');

			$c.data("highlight-zIndex", oldZ)
				.css('zIndex', z++)
				.addClass(movedClassName);

			$("<div>")
				.addClass(shadowClassName)
				.css('zIndex', z++)
				.appendTo($c);
		});

		this.data('highlight-zIndex', this.css('zIndex'))
			.css('zIndex', z++)
			.addClass(movedClassName);

		if(opts.placeAbove){
			var el = opts.placeAbove,
				pos = el.offset(),
				par = el.parent()[0];
			el.appendTo(parents.last())
				.offset(pos)
				.data('highlight-parent', par)
				.addClass(placedAboveClassName);
		}

		return this;
	};

	$.fn.unhighlight = function(){
		// Destroy all shadows
		$("."+shadowClassName).remove();

		// Change everything back to original stacking order
		$("."+movedClassName).each(function(i, container){
			var $c   = $(container),
				oldZ = $c.data("highlight-zIndex");
			$c.css('zIndex', oldZ).removeClass(movedClassName);
		});

		// Move and position item placed above shadows
		var placed = $("."+placedAboveClassName),
			offset = placed.offset(),
			par    = placed.data('highlight-parent');

		placed.appendTo(par).offset(offset);

		return this;
	};


	// Cover elements with an invisible layer to disable interaction
	//  Assumes that ghostClassName is styled to be full size and transparent
	$.fn.ghost = function(){
		try {
			$("<div>")
				.addClass(ghostClassName)
				.appendTo(this.find("*"))
				.bind("click mousedown", function(e){
					e.preventDefault();
					e.stopImmediatePropagation();
				});
		} catch(err) {}
		return this;
	};

	$.fn.unghost = function(){
		$("."+ ghostClassName).remove();
		return this;
	};
}));
