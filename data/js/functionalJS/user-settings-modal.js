/**
  * Creates a new user settings modal
  *
  * <p><b>Require Path:</b> shared/masthead/views/user-settings-modal</p>
  *
  * @module Shared
  * @submodule Shared.Masthead
  * @namespace Masthead
  * @class UserSettingsModal
  * @constructor
  * @extends foundation/views/base-view
 **/
define('shared/masthead/views/user-settings-modal',[
    'jquery/nyt',
    'underscore/nyt',
    'foundation/views/base-view',
    'shared/modal/views/modal',
    'shared/masthead/templates',
    'foundation/cookies'
], function ($, _, BaseView, Modal, templates, cookie) {
    'use strict';

    var UserSettingsModal = BaseView.extend({

        events: {
            'click .edition-menu a': 'handleEditionClick',
            'click .type-sizer-menu li': 'handleFontSizeClick',
            'click .help-menu a': 'handleHelpMenuClick'
        },

        nytEvents: {
            'nyt:fontsizer-initialize': 'handleFontSizeChange',
            'nyt:fontsizer-change': 'handleFontSizeChange'
        },

        defaultSettings: {
            id: 'user-settings-modal',
            binding: '.user-settings-button',
            tailDirection: 'up-right',
            tailTopOffset: -8,
            tailLeftOffset: -1,
            width: '260px',
            toggleSpeed: 1,
            showTypeSizer: true,
            openCallback: function () {
                this.$target.addClass('active');
                this.subscribeOnce('nyt:page-scroll', this.close);
            },
            closeCallback: function () {
                this.$target.removeClass('active');
            }
        },

        /**
         * Tracking parameter for the WT.nav key
         *
         * @static
         * @property webTrendsNav
         * @type String
         * @default 'user-settings-modalContainer'
        **/
        webTrendsNav: 'user-settings-modalContainer',

        /**
         * Initializes the settings modal
         *
         * @method constructor
        **/
        initialize: function (options) {
            var contentCollection = this.pageManager.getMeta('article:section') || '';

            this.settings = _.extend({}, this.defaultSettings, options);
            this.settings.modalContent = templates.userSettingsModal({
                showTypeSizer: this.settings.showTypeSizer
            });

            this.trackingBaseData = {
                'action': 'click',
                'region': 'TopBar',
                'WT.nav': this.webTrendsNav,
                'contentCollection': contentCollection
            };
        },

        /**
         * Renders the settings modal when the dom is ready
         *
         * @private
         * @method handleDomReady
        **/
        handleDomReady: function () {
            this.render();
        },

        /**
         * Renders the settings modal
         *
         * @private
         * @method render
        **/
        render: function () {
            var editionModal = new Modal(this.settings);

            //set the context of the view to the modal
            this.setElement(editionModal.$modal);

            //Toggle the selected edition in the menu
            editionModal.$modal.find('a[data-edition=' + this.pageManager.getEdition() + ']').addClass('selected');

            //Add modal to page
            editionModal.addToPage();

            //Update the font sizer button selection
            this.handleFontSizeChange();
        },

        /**
         * Sets the cookie for the us and global editions
         *
         * @private
         * @method handleEditionClick
        **/
        handleEditionClick: function (e) {
            this.broadcast('nyt:masthead-edition-change', e, { 'WT.nav': this.webTrendsNav });
        },

        /**
         * Handler for when user clicks on a help menu link
         *
         * @private
         * @method handleHelpMenuClick
         * @param {Object} e Click event object
        **/
        handleHelpMenuClick: function (e) {
            var $el = $(e.target);
            var href = $el.attr('href');

            if (href) {
                href = this.trackingAppendParams(href, {});
                $el.attr('href', href);
            }
        },

        /**
         * Handle font size clicks and tell the font sizer the correct size
         *
         * @private
         * @method handleEditionClick
        **/
        handleFontSizeClick: function (e) {
            var size = $(e.currentTarget).data('size');

            //tells font sizer to change the global settings
            this.broadcast('nyt:fontsizer-change', size);
        },

        /**
         * Dermines which font button to hightlight
         *
         * @private
         * @method selectFontSize
        **/
        handleFontSizeChange: function (size) {
            var className = 'type-size-selected';
            size = size || this.pageManager.getMeta('fontsizer_typeSize');
            this.$el.find('.type-sizer-menu li').removeClass(className);
            this.$el.find('.type-sizer-' + size).addClass(className);
        }

    });

    return UserSettingsModal;
});

