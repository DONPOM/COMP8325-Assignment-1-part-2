/**
 * Creates a new login modal on the site.
 *
 * <p><b>Require Path:</b> shared/account/views/login-modal</p>
 *
 * @module Shared
 * @submodule Shared.Account
 * @class LoginModal
 * @constructor
 * @extends foundation/views/base-view
**/
define('shared/account/views/login-modal',[
    'jquery/nyt',
    'underscore/nyt',
    'foundation/views/base-view',
    'foundation/models/user-data',
    'shared/modal/views/modal',
    'shared/account/templates',
    'foundation/lib/auth/userauth',
    'foundation/hosts',
    'shared/account/helpers/account-modal-mixin',
    'foundation/lib/polyfills/placeholder'
], function ($, _, BaseView, userData, Modal, templates, Auth, hosts, AccountModalMixin, placeholder) {
    'use strict';

    var LogInModal = BaseView.extend(

        _.extend({}, AccountModalMixin, {

            events: {
                'click .registration-modal-trigger': 'handleRegisterClick',
                'click #login-send-button': 'handleLogInButton',
                'submit #login-form': 'handleDefaultSubmit',
                'click .clear-button': 'handleClearButton',
                'keyup .text': 'handleKeyUp'
            },

            nytEvents: {
                'nyt:loginmodal-open': 'handleOpen'
            },

            defaultSettings: {
                id: 'login-modal',
                addlClasses: 'account-modal',
                modalContent: templates.logInModal({hosts: hosts}),
                binding: '.login-modal-trigger',
                tailDirection: 'fixed',
                hasCloseButton: true,
                hasOverlay: true,
                tailLeftOffset: -40, //accounts for padding
                tailTopOffset: -40 //accounts for padding
            },

            /**
             * Initializes the login in modal
             *
             * @method constructor
            **/
            initialize: function (options) {
                var loginView = this;

                this.modalSettings = _.extend({}, this.defaultSettings, options);
                this.modalSettings.openCallback = function() {
                    var title = loginView.modalTitle || this.$target.data('modal-title') || 'Log in';
                    //focus the cursor in the username box
                    if (loginView.hasPlaceholderSupport) {
                        this.$modal.find('.login-email').focus();
                    }
                    //add the ability to override the default title
                    this.$modal.find('legend').text(title);

                    //clear the inputs of any existing value
                    this.$modal.find('.field-container input').val('');

                    //clear the value of the property modalTitle
                    loginView.modalTitle = null;
                };

                this.subscribe('nyt:modal-show-login-modal', placeholder);
            },

            /**
             * Renders the login modal when the dom is ready
             *
             * @public
             * @method handleDomReady
            **/
            handleDomReady: function () {
                this.render();
            },

            /**
             * Creates a new login modal and adds it to the page
             *
             * @private
             * @method render
            **/
            render: function () {
                //create a new log in modal and add it to the page
                this.logInModal = new Modal(this.modalSettings).addToPage();

                //set the context of this view to the modal
                this.setElement(this.logInModal.$modal);

                //trigger event to re-check if modal should be removed
                this.local(this, 'nyt:modal-login-modal-rendered');
            },

            /**
             * Opens login modal based on an event
             *
             * @private
             * @method handleOpen
            **/
            handleOpen: function (options) {
                if (options && options.modalTitle) {
                    this.modalTitle = options.modalTitle;
                }

                if (this.logInModal) {
                    this.logInModal.open();
                }
            },

            /**
             * If the user-data model comes back as logged in, remove the modal
             *
             * @private
             * @method handleUserReady
            **/
            handleUserReady: function () {
                if (userData.isLoggedIn()) {
                    if (this.logInModal) {
                        this.logInModal.removeFromPage();
                        this.remove();
                    } else {
                        this.subscribeOnce(this, 'nyt:modal-login-modal-rendered', this.handleUserReady);
                    }
                }
            },

            /**
             * Closes login modal
             *
             * @private
             * @method handleRegisterClick
            **/
            handleRegisterClick: function () {
                this.logInModal.close();

                this.trackingTrigger('loginmodal-register-click', {
                    'module': 'Registration',
                    'action': 'Click',
                    'WT.z_ract': 'Regnow'
                });
            },

            /**
             * Handles the Log In button click and uses the Auth API to submit the form
             *
             * @private
             * @method handleLogInButton
            **/
            handleLogInButton: function () {
                this.executeLogin();
            },

            /**
             * Handles the Log In button click and uses the Auth API to submit the form
             *
             * @private
             * @method handleLogInButton
            **/
            executeLogin: function () {
                Auth.login({
                    user:       this.$el.find('.login-email').val(),
                    password:   this.$el.find('.login-password').val(),
                    remember:   this.$el.find('.login-remember').is(':checked')
                });
            }
        })
    );

    return LogInModal;
});

