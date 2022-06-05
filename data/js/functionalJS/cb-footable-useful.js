/* FooTable init code */

var $FOOTABLE = $FOOTABLE || {};
(function( $FOOTABLE, $, undefined ) {

	jQuery.fn.attrAppendWithComma=function(a,b){var c;return this.each(function(){c=$(this),void 0!==c.attr(a)&&""!=c.attr(a)?c.attr(a,c.attr(a)+","+b):c.attr(a,b)})};jQuery.fn.footableAttr=function(a,b){return this.each(function(){var c=$(this);c.data("auto-columns")!==!1&&(c.find("thead th:gt("+a+")").attrAppendWithComma("data-hide","tablet"),c.find("thead th:gt("+b+")").attrAppendWithComma("data-hide","phone"))})},jQuery.fn.footableFilter=function(a){return this.each(function(){var b=$(this);b.data("filter")||b.data("filter")===!1||b.data("filter-text-only","true").before('<div class="footable-filter-container"><input placeholder="'+a+'" style="float:right" type="text" class="footable-filter" /></div>')})},jQuery.fn.footablePager=function(){return this.each(function(){var a=$(this);if(a.data("page")!==!1){var b=$('<tfoot class="hide-if-no-paging"><tr><td><div class="pagination pagination-centered"></div></td></tr></tfoot>');b.find("td").attr("colspan",a.find("thead th").length),a.find("tbody:last").after(b)}})};

	$FOOTABLE.init = function() {
		$(".footable, .tablepress")
			.footableAttr(3,1)
			.footableFilter("search")
			.footablePager()
			.footable( { breakpoints: { phone: 320, tablet: 768 } });

	};
}( $FOOTABLE, jQuery ));

jQuery(function($) {
	$FOOTABLE.init();
});
