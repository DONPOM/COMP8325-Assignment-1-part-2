/**
  * Implements the Auto Suggest and Search functionality in the Masthead.
  *
  * <p><b>Require Path:</b> shared/masthead/views/search</p>
  *
  * @module Shared
  * @submodule Shared.Masthead.Search
  * @namespace Masthead.Search
  * @class View
  * @constructor
  * @extends foundation/views/base-view
 **/
define('shared/masthead/views/search',[
    'jquery/nyt',
    'underscore/nyt',
    'foundation/views/base-view',
    'shared/searchform/helpers/search-form-mixin'
], function ($, _, BaseView, SearchMixin) {
    'use strict';

    var SearchView = BaseView.extend(

        _.extend({}, SearchMixin, {

            el: $('#masthead').find('.search-flyout-panel'),

            //Using ids for keyup/down does not trigger events for lt-ie10
            events: {
                'click .close-button': 'close',
                'keyup .search-input': 'handleKeyUp',
                'keydown .search-input': 'handleKeyDown',
                'click .submit-button' : 'submitSearch',
                'click .clear-button': 'handleClearButton'
            },

            nytEvents: {
                'nyt:masthead-search-click': 'toggle',
                'nyt:masthead-search-toggle': 'handleSearchToggle',
                'nyt:navigation-search-click': 'toggle',
                'nyt:search-close-ready': 'close'
            },

            slideTiming: 100,

            /**
             * Add additional selectors once the search is ready
             *
             * @private
             * @method afterInitialize
            **/
            afterInitialize: function () {
                this.trackingBaseData['WT.nav'] = 'searchWidget';

                //no placeholder text for the masthead search
                this.$input.removeAttr('placeholder');
            },

            /**
             * Opens up the search flyout and wires up all the events.
             *
             * @method open
            **/
            open: function () {
                var searchObj = this;

                //slide down and fade in
                this.$el.slideDown(this.slideTiming, function () {
                    searchObj.$el
                        .find('.control').fadeIn(searchObj.slideTiming).end()
                        .find('.search-input').val('').focus();
                });

                //tells the masthead to set search button to active
                this.broadcast('nyt:masthead-search-toggle', true);

                //enable: hide auto suggest when you click off
                this.$shell.on('click', this.handlePageClicks);

                //enable: close window on escape
                this.subscribeOnce('nyt:page-key-esc', this.close);

                //show overlay
                this.$overlay.fadeIn(200);

                //add search-active class
                this.$html.addClass('search-active');
            },

            /**
             * Closes the search flyout and disconnects the related events.
             *
             * @method close
            **/
            close: function () {
                var searchObj = this;

                //determine if the search is currently active
                if(!this.$html.hasClass('search-active')) {
                    return;
                }

                //fade out and slide up
                this.$el.find('.control').fadeOut(this.slideTiming, function () {
                    searchObj.$el.slideUp(searchObj.slideTiming, function() {
                        searchObj.$html.removeClass('search-active');
                    });
                });

                //tells the masthead to set search button to in-active
                this.broadcast('nyt:masthead-search-toggle', false);

                //disable: hide auto suggest when you click off
                this.$shell.off('click', this.handlePageClicks);

                //delete the contents of your search on close
                this.setSearchValue('');

                //show overlay
                this.$overlay.fadeOut(200);
            },

            /**
             * Toggles the search flyout based on its current state.
             *
             * @method toggle
            **/
            toggle: function () {
                this[this.$el.is(':visible') ? 'close' : 'open']();
            },

            /**
             * Tell if the element is a search button
             *
             * @private
             * @method isSearchButton
             * @param {String} elementToCheck identifier for element that needs to be identified
             * @return {Boolean} whether the element is or isn't part of the search button
            **/
            isSearchButton: function (elementToCheck) {
                var isButton = $(elementToCheck).hasClass('search-button');
                var isButtonChild = $(elementToCheck).parents('.search-button').length === 1;
                return isButton || isButtonChild;
            },

            /**
             * Tell if the element is the search flyout
             *
             * @private
             * @method isSearchFlyout
             * @param {String} elementToCheck identifier for element that needs to be identified
             * @return {Boolean} whether the element is or isn't part of the search flyout
            **/
            isSearchFlyout: function (elementToCheck) {
                var isFlyout = $(elementToCheck).hasClass('search-flyout-panel');
                var isFlyoutChild = $(elementToCheck).parents('.search-flyout-panel').length === 1;
                return isFlyout || isFlyoutChild;
            },

            /**
             * Fired whenever there is a click on the document. Clicks outside the search
             * flyout will close the module. Clicks inside the search flyout will close auto-suggest
             *
             * @private
             * @method handlePageClicks
             * @param e {Object} document click event object
            **/
            handlePageClicks: function (e) {
                // search button opening will fire this binding, then be pick up the click and
                //close right after opening
                if (!this.isSearchButton(e.target) && !this.isSearchFlyout(e.target)) {
                    this.close();
                }

                //all clicks hide the auto suggest
                this.hideAutoSuggest();
            },

            /**
             * Reset the state of the search form when closing the menu
             *
             * @private
             * @method handleSearchToggle
             * @param isOpen {Boolean} open or close
            **/
            handleSearchToggle: function (isOpen) {
                if (isOpen === false) {
                    this.$el.find('.submit-button').addClass('disabled');
                    this.$el.find('.clear-button').hide();
                }
            }
        })
    );

    return SearchView;
});

