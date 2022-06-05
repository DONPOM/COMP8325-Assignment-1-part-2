/**
 * Creates an account mixin to minimize the duplication in account.
 *
 * <p><b>Require Path:</b> shared/account/views/account-mixin</p>
 *
 * @module Shared
 * @submodule Shared.Account
 * @namespace Account
 * @static AccountMixin
**/
define('shared/account/helpers/account-modal-mixin',[
    'jquery/nyt',
    'underscore/nyt',
    'foundation/lib/polyfills/placeholder'
], function ($, _, placeholder) {
    'use strict';

    var AccountModalMixin = {

        /**
         * Prevents the default submit action from occuring
         *
         * @private
         * @method handleFormSubmit
        **/
        handleDefaultSubmit: function (e) {
            e.preventDefault();
        },

        /**
         * Handles the keyup event of the input fields
         *
         * @private
         * @method handleKeyUp
        **/
        handleKeyUp: function (e) {
            //Issue in IE, don't display the clear button
            var $field = $(e.target);
            //toggles adjacent clear button
            $field.parent().find('.clear-button').toggle($field.val() !== '');
        },

        /**
         * Handles the click event of the input field clear event
         *
         * @private
         * @method handleClearButton
        **/
        handleClearButton: function (e) {
            var $clearButton = $(e.currentTarget);
            var $field = $clearButton.parent().find('input');

            $clearButton.toggle(false);
            $field.val('');
            placeholder();
        }
    };

    return AccountModalMixin;

});
