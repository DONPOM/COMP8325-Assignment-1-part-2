/**
 * Creates a new account modal
 *
 * <p><b>Require Path:</b> shared/masthead/views/account-modal</p>
 *
 * @module Shared
 * @submodule Shared.Masthead
 * @namespace Masthead
 * @class UserNameModal
 * @constructor
 * @extends foundation/views/base-view
**/
define('shared/masthead/views/user-name-modal',[
    'jquery/nyt',
    'underscore/nyt',
    'foundation/views/base-view',
    'shared/modal/views/modal',
    'shared/masthead/templates',
    'foundation/models/user-data',
    'foundation/hosts'
], function ($, _, BaseView, Modal, templates, userData, hosts) {
    'use strict';

    var UserNameModal = BaseView.extend({

        events: {
            'click .premier-subscriber a': 'handleUserSubscriptionClick',
            'click li a': 'handleUserOptionClick'
        },

        defaultSettings: {
            id: 'user-name-modal',
            modalTitle: '',
            modalContent: '',
            binding: '.user-name-button',
            tailDirection: 'up-right',
            tailTopOffset: -8,
            tailLeftOffset: -1,
            width: '260px',
            toggleSpeed: 1,
            openCallback: function () {
                this.$target.addClass('active');
                this.listenToOnce(this.pageManager, 'nyt:page-scroll', this.close);
            },
            closeCallback: function () {
                this.$target.removeClass('active');
            }
        },

        /**
         * Initializes the user name modal
         *
         * @method constructor
        **/
        initialize: function (options) {
            _.bindAll(this, 'handleLogOut');

            this.settings = _.extend({}, this.defaultSettings, options);
            this.trackingBaseData = {
                'module': 'ProfileDropdown',
                'action': 'click',
                'region': 'Masthead',
                'WT.nav': 'shell'
            };
        },

        /**
         * Renders the user name modal when the user is ready
         *
         * @private
         * @method handleUserReady
        **/
        handleUserReady: function () {
            var $subscription;
            var subscription = this.getSubscriptionLabel(userData);

            this.settings.modalTitle = templates.userNameModalTitle({
                userData: userData,
                subscription: subscription
            });

            this.settings.modalContent = templates.userNameModalContent({
                userData: userData,
                hosts: hosts
            });

            this.render();

            $subscription = this.$el.find('.user-subscription');
            if (subscription.label === '') {
                $subscription.remove();
            }

            if (subscription.className !== '') {
                $subscription.addClass(subscription.className);
            }

            if (subscription.link !== '') {
                $subscription.find('a').attr("href", subscription.link);
            }

            if (subscription.label === 'NYT Now Subscriber') {
                $subscription.find('a').after('<i class="icon dot-logo-icon"></i>');
            }

            this.listenToOnce(userData, 'nyt:user-image-loaded', this.handleProfileImage);
            this.$el.on('click', '.log-out-button', this.handleLogOut);
        },

        /**
         *  Returns an object for displaying subscription status in the dropdown
         *
         * @private
         * @method getSubscriptionLabel
        **/
        getSubscriptionLabel: function (userData) {
            var subscription = {
                label: '',
                className: '',
                link: ''
            };
            if (userData.isPremierSubscriber()) {
                subscription.label = 'Times Premier Subscriber';
                subscription.className = 'premier-subscriber';
                subscription.link = 'http://www.nytimes.com/times-insider';
            } else if (userData.isWebSubscriber() && userData.isMobileSubscriber()) {
                subscription.label = 'All Digital Access Subscriber';
            } else if (userData.isTabletSubscriber() && userData.isWebSubscriber()) {
                subscription.label = 'Tablet and Web Subscriber';
            } else if (userData.isSmartphoneSubscriber() && userData.isWebSubscriber()) {
                subscription.label = 'Smartphone and Web Subscriber';
            } else if (userData.isWebSubscriber()) {
                subscription.label = 'Web Subscriber';
            } else if (userData.isNytNowSubscriber() && userData.isCookingSubscriber()) {
                subscription.label = 'NYT Now & Cooking Subscriber';
                subscription.link = 'http://www.nytnow.com';
            } else if (userData.isNytNowSubscriber() && userData.isOpinionSubscriber()) {
                subscription.label = 'NYT Now & NYT Opinion Subscriber';
                subscription.link = 'http://www.nytimes.com/content/help/mobile/mobile.html';
            } else if (userData.isNytNowSubscriber()) {
                subscription.label = 'NYT Now Subscriber';
                subscription.link = 'http://www.nytimes.com/content/help/mobile/nyt-now/nyt-now.html';
            } else if (userData.isOpinionSubscriber() && userData.isCookingSubscriber()) {
                subscription.label = 'NYT Opinion & Cooking Subscriber';
            } else if (userData.isCookingSubscriber()) {
                subscription.label = 'Cooking Subscriber';
            } else if (userData.isOpinionSubscriber()) {
                subscription.label = 'NYT Opinion Subscriber';
                subscription.link  = 'http://www.nytimes.com/content/help/mobile/nyt-opinion/nyt-opinion.html';
            } else if (userData.isCrosswordsSubscriber()) {
                subscription.label = 'Crosswords Subscriber';
            }
            return subscription;
        },

        /**
         * Renders the account modal
         *
         * @private
         * @method render
        **/
        render: function () {
            var userNameModal = new Modal(this.settings);

            //set the context of the view to the modal
            this.setElement(userNameModal.$modal);

            //adds the account modal to the page
            userNameModal.addToPage();
        },

        /**
         * Handles the async loading of the profile image
         *
         * @private
         * @method handleProfileImage
        **/
        handleProfileImage: function () {
            var html = templates.profileIcon({userData: userData});
            this.$el.find('.modal-heading').prepend(html);
        },

        handleUserSubscriptionClick: function (e) {
            var $el = $(e.currentTarget);
            var href = this.trackingAppendParams($el.attr('href'), {
                'contentCollection': 'TimesPremier-InsideStory'
            });
            $el.attr('href', href);
        },

        /**
         * Handles user clicking on any option in the username modal
         *
         * @private
         * @method handleUserOptionClick
         * @param {Object} e Click event object
        **/
        handleUserOptionClick: function (e) {
            var $target = $(e.target);
            var text = $target.text();
            var href = $target.attr('href');
            var data = {
                'action': 'Click',
                'region': 'TopBar',
                'WT.nav': 'shell'
            };

            if (href) {
                if (text) {
                    data.module = text.replace(/\s/g, '');
                }

                href = this.trackingAppendParams(href, data);
                $target.attr('href', href);
            }
        },

        /**
         * Handles the logout behavior
         *
         * @private
         * @method handleProfileImage
        **/
        handleLogOut: function () {
            var href = 'http:' + hosts.www + '/logout';

            href = this.trackingAppendParams(href, {
                'module': 'LogOut',
                'action': 'Click',
                'region': 'TopBar',
                'WT.nav': 'shell'
            });

            window.location = href;
        }

    });

    return UserNameModal;
});

