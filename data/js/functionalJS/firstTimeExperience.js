// First Time Experience
//
//
define([
	'jquery',
	'spine',
	'internal/sitebuilder/builderChrome/toolbar',
	'internal/sitebuilder/builderChrome/chromeController',
	'jquery.highlight',
	'jquery.jplayer',
	'handlebars'
], function($, Spine, toolbar){

	// Simple controller to walk through a set of steps and emit events
	var Walkthrough = Spine.Class.create({
		init: function(opts){
			this.steps  = opts.steps;
			this.stepId = 0;
		},

		step: function(){
			return this.steps[this.stepId];
		},

		hasNext: function(){
			return this.stepId + 1 < this.steps.length;
		},

		hasPrev: function(){
			return this.stepId > 0;
		},

		next: function(){
			if(!this.hasNext()){
				this.trigger("nextFailed");
				return false;
			} else {
				this.trigger("beforeNext");
				this.stepId += 1;
				this.trigger("next");
				return this.stepId;
			}
		},

		prev: function(){
			if(!this.hasPrev()){
				this.trigger("prevFailed");
				return false;
			} else {
				this.trigger("beforePrev");
				this.stepId -= 1;
				this.trigger("prev");
				return this.stepId;
			}
		}
	});
	Walkthrough.include(Spine.Events);

	// Steps in the first-time experience.
	// content: html to be displayed in the dialog
	// arrow: class to be applied to the arrow div
	// highlight: a list of selectors to bring above the shadow
	var firstTimeStepDefaults = {
			title: "SiteBuilder Tour",
			content: "",
			nextButtonText: "Next Step",
			arrow: false,
			arrowPosition: { top: 0, left: 0 },
			highlight: false,
			height: 180
		},

		irItems = {
			panelText: function(){
				if (webs.useIR){
					return 'Dashboard';
				} else {
					return 'ControlPanel';
				}
			},
			arrowPosY: function(){
				if (webs.useIR){
					return 51;
				} else {
					return 41;
				}
			}
		},

		firstTimeSteps = [
			{
				title: "Welcome to SiteBuilder!",
				content: "<h2>You have now entered our super easy-to-use SiteBuilder.</h2><h3>Let's take a quick tour of how it works.</h3>",
				nextButtonText: "Start Tour",
				height: 150,
				highlight: "#webslogo, .global-header .logo"
			},
			{
				content: ("<h2>Simply Drag & Drop<small>(Step 1 of 5)</small></h2>" +
				          "<div style='display: inline-block; width: 220px; vertical-align: top; padding: 10px;'>" +
				              "<p>The Dock contains all the building blocks needed for your website.<br/><br/>" +
				              "To add content to your site, you simply <strong>drag &amp; drop</strong> modules (text, images, videos, etc.) into the main content area of your site.</p>" +
				          "</div>" +
				          "<div style='display: inline-block; width: 320px'>" +
				          	 "<div id='fte_module_video'></div>" +
				          "</div>"),
				arrow: "moduleBar",
				arrowPosition: { top: -350, left: 10 },
				height: 350,
				width: 600,
				reposition: true,
				highlight: "#leaf_dock",
				activated: function(){
					$("#controls_wrap").css('zIndex', 99999999999);
					$("#fte_module_video").jPlayer({
						size: {width: 320, height: 240},
						loop: true,
						swfPath: webs.props.globalAssetBaseURL + '/static/global/js/jquery/plugins/jplayer2/Jplayer.swf',
						supplied: "m4v, ogv, flv, poster",
						solution:"flash, html",
						ready: function() {
							$("#fte_module_video").
								jPlayer("setMedia", {
									"poster": webs.props.globalAssetBaseURL + '/static/projects/finch/fte-videos/SB3-modules.jpg',
									"m4v": webs.props.globalAssetBaseURL + '/static/projects/finch/fte-videos/SB3-modules.m4v',
									"ogv": webs.props.globalAssetBaseURL + '/static/projects/finch/fte-videos/SB3-modules.ogv',
									"flv": webs.props.globalAssetBaseURL + '/static/projects/finch/fte-videos/SB3-modules.flv'
								}).
								jPlayer("play");
						}
					});
				},
				deactivated: function(){
					$("#controls_wrap").css('zIndex', '');
					$("#fte_module_video").jPlayer("destroy");
				}
			},
			{
				content: "<h2>The Editing Bar<small>(Step 2 of 5)</small></h2><p> After dragging an element onto your page, you can click on it to make edits.  This bar will then appear allowing you to customize things like font formatting, colors, picture frames, and other settings.</p>",
				arrow: "editBar",
				arrowPosition: { top: 70, left: 80 },
				highlight: "#module_toolbar",
				height: 175,
				activated: function(context){
					context.toolbar = toolbar.create('text');
					context.toolbar.show({});
					$('header.global-header').css('zIndex',10000010);
				},
				deactivated: function(context){
					toolbar.switchToolbar();
					$('header.global-header').css('zIndex','');
				}
			},
			{
				content: "<h2>Page Settings<small>(Step 3 of 5)</small></h2><p>From this tab you can change page-specific settings like the page name, layout, password protection, etc.</p>",
				arrow: "undoRedo",
				arrowPosition: { top: 30, left: -325 },
				highlight: "#right_tools",
				height: 175,
				activated: function(context){
					$("#page_options").addClass("active");
					$('header.global-header').css('zIndex',10000010);
				},
				deactivated: function(context){
					$("#page_options").removeClass("active");
					$('header.global-header').css('zIndex','');
				}
			},
			{
				content: "<h2>Main Navigation<small>(Step 4 of 5)</small></h2>"+
							 "<p><strong>Theme</strong></p>"+
							 "<p>" +
								"Change the look and feel of your entire site by switching templates " +
								"or by customizing your site&rsquo;s colors, backgrounds and fonts."+
							 "</p>" +
							 "<p><strong>Pages</strong></p>"+
							 "<p>Add, hide, remove and re-order the pages on your site.</p>"+
							 "<p class='controlPanel'><strong>" + irItems.panelText() + "</strong></p>"+
							 "<p class='controlPanel'>A centralized spot where you can access all of your other site management tools.</p>",
				arrow: "mainNav",
				arrowPosition: { top: irItems.arrowPosY(), left: 150 },
				height: 290,
				highlight: "#w-main-nav-menu, .header-nav",
				activated: function(context){
					$("#w-main-nav-menu, .header-nav").css('position', 'relative');
					if(!webs.site.showControlPanel){
						$(".controlPanel").hide();
						//websover.resize({width: 480, height: 232 });
					}
				}
			},
			{
				content: "<h2>Publish<small>(Step 5 of 5)</small></h2><p>SiteBuilder auto-saves as you work.  When you're ready, click the  &ldquo;Publish&rdquo; button and any changes you&rsquo;ve made to your site will be made live on the internet.</p>",
				height: 165,
				arrow: "publish",
				arrowPosition: { top: irItems.arrowPosY(), left: -140 },
				highlight: "#bldr_publish, #publish",
				activated: function(context){
					$("#bldr_publish, #publish").css({position: 'relative', cursor: "default" });
				},
				deactivated: function(context){
					$("#bldr_publish, #publish").css({position: '', cursor: "" });
				}
			},
			{
				lastStep: true,
				shadow: false
			}
		];


	firstTimeSteps = $.map(firstTimeSteps, function(step, id){
		return $.extend({}, firstTimeStepDefaults, step, { id: id});
	});

	// location of image file for arrow name
	function arrowLocation(arrow){
		return webs.props.staticServer +
			"/static/projects/finch/images/firstTimeExperience/" +
			arrow +
			".png";
	}

	var FirstTimeExperience = Spine.Controller.create({
		events: {
			"click .next": "clickedNext",
			"click .prev": "clickedPrev"
		},
		proxied: ["next", "prev", "clickedNext", "clickedPrev", "popoverClosed", "undisplayStep"],

		init: function(opts){
			if(!opts.el) this.testing = true;
			if(!opts.steps) opts.steps = firstTimeSteps;
			this.walkthrough = Walkthrough.init({steps: opts.steps});

			if(typeof websover !== 'undefined') websover.animate = false;

			this.displayStep();
			this.walkthrough
				.bind("next", this.next)
				.bind("prev", this.prev)
				.bind("beforePrev beforeNext", this.undisplayStep);
			toolbar.switchToolbar();

			// Don't propagate dialog events any higher, so we don't change the UI in response to them.
			this.el.bind("click mouseup mousedown", function(){ return false; });
		},

		next: function(){
			this.displayStep();
		},

		prev: function(){
			this.displayStep();
		},

		clickedNext: function(){
			this.walkthrough.next();
		},

		clickedPrev: function(){
			this.walkthrough.prev();
		},

		undisplayStep: function(){
			var step = this.walkthrough.step();
			if(step.deactivated) step.deactivated(this);
			$.fn.unhighlight().unghost();
		},

		displayStep: function(){
			var step = this.walkthrough.step();
			this.el.html(this.template(step));
			this.showInPopover();
		},

		template: function(step){
			this.parent.fn.template = Handlebars.compile(this._template);
			return this.template(step);
		},

		_template:
			"<div class='w-fte-dialog'>" +
				"<div id='websover_subheader'></div>" +
				"<div class='contentblock step-{{id}}'>" +
					"{{{ content }}}" +
					"<div class='buttons'>" +
						"<a class='w-btn prev gray'><span><em>&nbsp;</em></span></a>" +
						"<a class='w-btn next'><span>{{nextButtonText}} <em class='arrow'>&nbsp;</em></span></a>" +
					"</div>" +
				"</div>" +
			"</div>",

		showInPopover: function(){
			if(this.testing || typeof Popover == "undefined") return false;
			var step = this.walkthrough.step(),
				width = step.width || 480;

			if(!this.popover) {
				this.popover = new Popover(this.el[0], {
					onClose: this.popoverClosed,
					width: width,
					height: step.height,
					heading: step.title,
					feedback: { ref: $("#help-options") }
				}).show();

				// hide popover's shadow and use our own
				$("#websover_shadow").hide();
				this.$websoverBorder = $("#websover_border");
				this.savedPosition = this.$websoverBorder.offset();
			} else {
				if (!this.savedPosition)
					this.savedPosition = this.$websoverBorder.offset();
				websover.resize({width: width, height: step.height, heading: step.title });
			}
			this.$websoverBorder.width(width);
			this.$websoverBorder.css({overflow: "visible", opacity: 0}).animate({ opacity: 1 });

			$('#publish').on('click', disableEvents);


			if(step.lastStep) return this.showLastStep(step);
			if(step.activated) step.activated(this);
			if(step.highlight) $(step.highlight).highlight({placeAbove: $("#websover_border").stop()}).ghost();
			if (step.reposition) {
				this.$websoverBorder.css({
					position: "fixed",
					left: parseInt(($(window).width() - width) / 2, 10) + "px",
					top: parseInt(($(window).height() - step.height - websover.extraHeight - 100) / 2, 10) + "px"
				});
			} else {
				this.$websoverBorder.css({
					position: "fixed",
					left: this.savedPosition.left + "px",
					top: this.savedPosition.top + "px"
				});

			}
			if(step.arrow){
				var
					highlightOffset = $(step.highlight).offset(),
					tourOffset = $('#websover_border').offset(),
					left = highlightOffset.left - tourOffset.left + step.arrowPosition.left,
					top = highlightOffset.top - tourOffset.top + step.arrowPosition.top;

				var arrowImg = $("<img/>").attr("src", arrowLocation(step.arrow))
									.addClass("arrowImg")
									.css({"left": left, "top": top})
									.appendTo(".w-fte-dialog");
			}
			$("#right_tools").ghost();
			$(".jq-highlight-shadow").bind("click mousedown mouseup", function(){ return false; });
		},

		showLastStep: function(step){
			this.$websoverBorder.css({"position": ""});
			if(typeof websover !== 'undefined') {
				websover.animate = true;
				setTimeout(function() {
					websover.animate = false;
					if (bldr.pageController.trees.find("webs-body").model.hasSinglePlaceholder()) {
						require("internal/sitebuilder/builderChrome/chromeController").showToasterHelptip();
					}
				}, 500);
			}
			this.popover.hide();
		},

		popoverClosed: function(notInstCall){
			this.undisplayStep();
			this.$websoverBorder.css({"position": ""});
			this.$("#websover_wrap").hide();

			$('#publish').off('click', disableEvents);

			// if not coming from an instance call, there's no feedback animation
			// this will force a feedback animation when clicking the dialog "x" icon
			if(notInstCall && this.popover) {
				websover.animate = true;
				setTimeout(function() {
					websover.animate = false;
					if (bldr.pageController.trees.find("webs-body").model.hasSinglePlaceholder()) {
						require("internal/sitebuilder/builderChrome/chromeController").showToasterHelptip();
					}
				}, 500);
				this.popover.hide();
			}

		}

	});

	var disableEvents = function(event){
		event.preventDefault();
		event.stopImmediatePropagation();
	};

	return FirstTimeExperience;

});
