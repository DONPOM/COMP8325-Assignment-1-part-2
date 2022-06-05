/**
 * Creates a new registration modal on the site.
 *
 * <p><b>Require Path:</b> shared/account/views/registration-modal</p>
 *
 * @module Shared
 * @submodule Shared.Account
 * @class RegistrationModal
 * @constructor
 * @extends foundation/views/base-view
**/
define('shared/account/views/registration-modal',[
    'jquery/nyt',
    'underscore/nyt',
    'foundation/views/base-view',
    'shared/modal/views/modal',
    'shared/account/templates',
    'foundation/lib/auth/userauth',
    'shared/account/helpers/account-modal-mixin',
    'foundation/lib/polyfills/placeholder'
], function ($, _, BaseView, Modal, templates, Auth, AccountModalMixin, placeholder) {
    'use strict';

    var RegistrationModal = BaseView.extend(

        _.extend({}, AccountModalMixin, {

            events: {
                'click .login-modal-trigger': 'handleLoginClick',
                'click #register-send-button': 'handleRegistrationButton',
                'submit #registration-form': 'handleDefaultSubmit',
                'click .clear-button': 'handleClearButton',
                'keyup .text': 'handleKeyUp'
            },

            defaultSettings: {
                id: 'registration-modal',
                addlClasses: 'account-modal',
                modalContent: templates.registrationModal(),
                binding: '.registration-modal-trigger',
                tailDirection: 'fixed',
                hasCloseButton: true,
                hasOverlay: true,
                tailLeftOffset: -40, //accounts for padding
                tailTopOffset: -40 //accounts for padding
            },

            /**
             * Initializes the registration modal
             *
             * @method constructor
            **/
            initialize: function (options) {
                var regiView = this;

                this.modalSettings = _.extend({}, this.defaultSettings, options);
                this.modalSettings.openCallback = function() {
                    var title = this.$target.data('modal-title') || 'Register';

                    //focus the cursor in the username box
                    if (regiView.hasPlaceholderSupport) {
                        this.$modal.find('.register-email').focus();
                    }

                    //add the ability to override the default title
                    this.$modal.find('legend').text(title);

                    //clear the inputs of any existing value
                    this.$modal.find('.field-container input').val('');
                };

                this.subscribe('nyt:modal-show-registration-modal', placeholder);
            },

            /**
             * Renders the regi modal when the dom is ready
             *
             * @public
             * @method handleDomReady
            **/
            handleDomReady: function () {
                this.render();
            },

            /**
             * Creates a new registration modal and adds it to the page
             *
             * @private
             * @method render
            **/
            render: function () {
                //create a new log in modal and add it to the page
                this.registrationModal = new Modal(this.modalSettings).addToPage();

                //set the context of this view to the modal
                this.setElement(this.registrationModal.$modal);
            },

            /**
             * Closes registration modal
             *
             * @private
             * @method handleRegisterClick
            **/
            handleLoginClick: function () {
                this.registrationModal.close();
            },

            /**
             * Handles the Registration button click and uses the Auth API to submit the form
             *
             * @private
             * @method handleRegistrationButton
            **/
            handleRegistrationButton: function () {
                this.executeRegistration();
            },

            /**
             * Handles the Registration button click and uses the Auth API to submit the form
             *
             * @private
             * @method handleLogInButton
            **/
            executeRegistration: function () {
                Auth.register({
                    email:          this.$el.find('.register-email').val(),
                    password:       this.$el.find('.register-password').val(),
                    confPassword:   this.$el.find('.retype-password').val(),
                    specialOffers:  this.$el.find('.receive-offers').is(':checked')
                });
            }
        })
    );

    return RegistrationModal;
});

