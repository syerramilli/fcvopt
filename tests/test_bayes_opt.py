#!/usr/bin/env python3
"""
Unit tests for BayesOpt class.

Tests the core functionality, refactored methods, and MLflow integration
of the BayesOpt optimizer.
"""

import unittest
import tempfile
import shutil
import os
import time
import numpy as np
import torch
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fcvopt.optimizers.bayes_opt import BayesOpt
from fcvopt.configspace import ConfigurationSpace
import ConfigSpace as CS


class TestBayesOpt(unittest.TestCase):
    """Test suite for BayesOpt class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple 2D optimization problem
        self.config_space = ConfigurationSpace()
        self.config_space.add(CS.Float('x', bounds=(-5.0, 5.0)))
        self.config_space.add(CS.Float('y', bounds=(-5.0, 5.0)))
        self.config_space.generate_indices()

        # Simple quadratic objective function (minimize x^2 + y^2)
        def objective(config):
            return config['x']**2 + config['y']**2

        self.objective = objective

        # Create temporary directory for MLflow tracking
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_init_basic(self):
        """Test basic initialization of BayesOpt."""
        bo = BayesOpt(
            obj=self.objective,
            config=self.config_space,
            minimize=True,
            tracking_dir=self.temp_dir
        )

        self.assertIsNotNone(bo)
        self.assertEqual(bo.minimize, True)
        self.assertEqual(bo.sign_mul, -1)
        self.assertEqual(bo.acq_function, 'EI')
        self.assertFalse(bo.batch_acquisition)
        self.assertIsNone(bo.train_confs)
        self.assertFalse(bo._mlflow_initialized)

    def test_init_maximize(self):
        """Test initialization with maximize=True."""
        bo = BayesOpt(
            obj=self.objective,
            config=self.config_space,
            minimize=False,
            tracking_dir=self.temp_dir
        )

        self.assertEqual(bo.minimize, False)
        self.assertEqual(bo.sign_mul, 1)

    def test_init_different_acq_functions(self):
        """Test initialization with different acquisition functions."""
        for acq_func in ['EI', 'LCB', 'KG']:
            with self.subTest(acq_func=acq_func):
                bo = BayesOpt(
                    obj=self.objective,
                    config=self.config_space,
                    acq_function=acq_func,
                    tracking_dir=self.temp_dir
                )
                self.assertEqual(bo.acq_function, acq_func)

    def test_init_batch_acquisition(self):
        """Test initialization with batch acquisition."""
        bo = BayesOpt(
            obj=self.objective,
            config=self.config_space,
            batch_acquisition=True,
            acquisition_q=3,
            tracking_dir=self.temp_dir
        )

        self.assertTrue(bo.batch_acquisition)
        self.assertEqual(bo.acquisition_q, 3)

    @patch('mlflow.start_run')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    def test_run_basic(self, mock_set_exp, mock_set_uri, mock_start_run):
        """Test basic run functionality."""
        # Mock MLflow
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_run.info.experiment_id = 'test_exp_id'
        mock_start_run.return_value.__enter__.return_value = mock_run

        bo = BayesOpt(
            obj=self.objective,
            config=self.config_space,
            tracking_dir=self.temp_dir,
            verbose=0  # Suppress output during tests
        )

        # Run for 3 iterations with 2 initial points
        results = bo.run(n_iter=3, n_init=2)

        # Check results structure
        self.assertIn('conf_inc', results)
        self.assertIn('f_inc_obs', results)
        self.assertIn('f_inc_est', results)

        # Check that we have trained on the expected number of points
        # 2 initial + 3 iterations = 5 total evaluations
        self.assertEqual(len(bo.train_confs), 5)
        self.assertEqual(bo.train_x.shape[0], 5)
        self.assertEqual(bo.train_y.shape[0], 5)

        # Check that incumbent is reasonable (should be close to [0, 0])
        incumbent_value = results['f_inc_obs']
        self.assertIsInstance(incumbent_value, float)
        self.assertGreaterEqual(incumbent_value, 0)  # x^2 + y^2 >= 0

    @patch('mlflow.start_run')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    def test_optimize_method(self, mock_set_exp, mock_set_uri, mock_start_run):
        """Test the optimize method interface."""
        # Mock MLflow
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_run.info.experiment_id = 'test_exp_id'
        mock_start_run.return_value.__enter__.return_value = mock_run

        bo = BayesOpt(
            obj=self.objective,
            config=self.config_space,
            tracking_dir=self.temp_dir,
            verbose=0
        )

        # Test initial run
        results = bo.optimize(n_trials=10, n_init=3)
        self.assertEqual(len(bo.train_confs), 10)

        # Test continuation run
        results2 = bo.optimize(n_trials=5)
        self.assertEqual(len(bo.train_confs), 15)

    def test_create_acquisition_function(self):
        """Test the new _create_acquisition_function method."""
        bo = BayesOpt(
            obj=self.objective,
            config=self.config_space,
            tracking_dir=self.temp_dir
        )

        # Initialize with some dummy data to create a model
        bo.train_x = torch.tensor([[0.0, 0.0], [1.0, 1.0]]).double()
        bo.train_y = torch.tensor([0.0, 2.0]).double()
        bo.model = bo._construct_model()
        bo.curr_f_inc_est = 0.0

        # Test EI acquisition function
        bo.acq_function = 'EI'
        acq_func = bo._create_acquisition_function()
        self.assertIsNotNone(acq_func)

        # Test LCB acquisition function
        bo.acq_function = 'LCB'
        acq_func = bo._create_acquisition_function()
        self.assertIsNotNone(acq_func)

        # Test KG acquisition function
        bo.acq_function = 'KG'
        acq_func = bo._create_acquisition_function()
        self.assertIsNotNone(acq_func)

        # Test invalid acquisition function
        bo.acq_function = 'INVALID'
        with self.assertRaises(ValueError):
            bo._create_acquisition_function()

    def test_select_next_candidates(self):
        """Test the new _select_next_candidates method."""
        bo = BayesOpt(
            obj=self.objective,
            config=self.config_space,
            tracking_dir=self.temp_dir
        )

        # Initialize with some dummy data
        bo.train_x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]]).double()
        bo.train_y = torch.tensor([0.0, 2.0, 2.0]).double()
        bo.model = bo._construct_model()
        bo.curr_f_inc_est = 0.0

        # Mock the acquisition optimization to return a simple result
        with patch('fcvopt.optimizers.bayes_opt._optimize_botorch_acqf') as mock_opt:
            mock_opt.return_value = (torch.tensor([[0.5, 0.5]]).double(), torch.tensor([1.0]))

            candidates = bo._select_next_candidates(0)

            self.assertIsInstance(candidates, list)
            self.assertEqual(len(candidates), 1)
            self.assertIsNotNone(bo.curr_acq_val)
            self.assertIsNotNone(bo.curr_acq_opt_time)

    def test_format_candidate_configs(self):
        """Test the new _format_candidate_configs method."""
        bo = BayesOpt(
            obj=self.objective,
            config=self.config_space,
            tracking_dir=self.temp_dir
        )

        # Create some dummy candidates
        conf1 = self.config_space.sample_configuration()
        conf2 = self.config_space.sample_configuration()
        bo.curr_conf_cand = [conf1, conf2]

        formatted = bo._format_candidate_configs()

        self.assertIsInstance(formatted, list)
        self.assertEqual(len(formatted), 2)
        self.assertIsInstance(formatted[0], dict)
        self.assertIsInstance(formatted[1], dict)

        # Test with no candidates
        bo.curr_conf_cand = None
        formatted = bo._format_candidate_configs()
        self.assertEqual(formatted, [])

    @patch('mlflow.start_run')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    @patch('mlflow.log_dict')
    def test_log_eval_with_kwargs(self, mock_log_dict, mock_set_exp, mock_set_uri, mock_start_run):
        """Test the enhanced _log_eval method with kwargs."""
        # Mock MLflow
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_run.info.experiment_id = 'test_exp_id'
        mock_start_run.return_value.__enter__.return_value = mock_run

        bo = BayesOpt(
            obj=self.objective,
            config=self.config_space,
            tracking_dir=self.temp_dir
        )

        # Initialize MLflow
        bo._initialize_mlflow()

        # Create test data
        conf = self.config_space.sample_configuration()
        x = conf.get_array()
        y = 1.5
        eval_time = 0.1

        # Test basic logging
        bo._log_eval(conf, x, y, eval_time)

        # Verify log_dict was called
        self.assertTrue(mock_log_dict.called)
        call_args = mock_log_dict.call_args[1]
        logged_data = mock_log_dict.call_args[0][0]

        self.assertIn('idx', logged_data)
        self.assertIn('conf', logged_data)
        self.assertIn('x', logged_data)
        self.assertIn('y', logged_data)
        self.assertIn('eval_time', logged_data)

        # Test logging with additional kwargs
        bo._log_eval(conf, x, y, eval_time, fold_idx=3, custom_metric=42.0)

        # Get the latest call
        latest_call = mock_log_dict.call_args_list[-1]
        logged_data = latest_call[0][0]

        self.assertIn('fold_idx', logged_data)
        self.assertIn('custom_metric', logged_data)
        self.assertEqual(logged_data['fold_idx'], 3)
        self.assertEqual(logged_data['custom_metric'], 42.0)

    def test_context_manager(self):
        """Test BayesOpt as a context manager."""
        with patch('mlflow.start_run'), patch('mlflow.set_tracking_uri'), \
             patch('mlflow.set_experiment'), patch('mlflow.end_run') as mock_end_run:

            with BayesOpt(obj=self.objective, config=self.config_space,
                         tracking_dir=self.temp_dir) as bo:
                self.assertIsNotNone(bo)

            # Verify end_run was called when exiting context
            mock_end_run.assert_called_once()

    def test_get_optimization_results_error_cases(self):
        """Test error cases for get_optimization_results."""
        bo = BayesOpt(
            obj=self.objective,
            config=self.config_space,
            tracking_dir=self.temp_dir
        )

        # Test error when no optimization has been performed
        with self.assertRaises(RuntimeError):
            bo.get_optimization_results()

    def test_invalid_acquisition_function(self):
        """Test initialization with invalid acquisition function."""
        with self.assertRaises(ValueError):
            bo = BayesOpt(
                obj=self.objective,
                config=self.config_space,
                acq_function='INVALID_ACQ',
                tracking_dir=self.temp_dir
            )
            # Initialize to trigger the error
            bo.train_x = torch.tensor([[0.0, 0.0]]).double()
            bo.train_y = torch.tensor([0.0]).double()
            bo.model = bo._construct_model()
            bo.curr_f_inc_est = 0.0
            bo._create_acquisition_function()

    def test_evaluate_single_config(self):
        """Test evaluation of a single configuration."""
        bo = BayesOpt(
            obj=self.objective,
            config=self.config_space,
            tracking_dir=self.temp_dir
        )

        conf = self.config_space.sample_configuration()
        x, y, eval_time = bo._evaluate(conf)

        self.assertEqual(x.shape, (2,))  # 2D configuration space
        self.assertIsInstance(y, (int, float))
        self.assertIsInstance(eval_time, (int, float))
        self.assertGreater(eval_time, 0)

        # Check that the evaluation matches our objective function
        expected_y = conf['x']**2 + conf['y']**2
        self.assertAlmostEqual(y, expected_y, places=10)

    def test_evaluate_multiple_configs_parallel(self):
        """Test parallel evaluation of multiple configurations."""
        bo = BayesOpt(
            obj=self.objective,
            config=self.config_space,
            n_jobs=2,  # Enable parallel evaluation
            tracking_dir=self.temp_dir
        )

        confs = [self.config_space.sample_configuration() for _ in range(3)]
        results = bo._evaluate_confs(confs)

        self.assertEqual(len(results), 3)
        for x, y, eval_time in results:
            self.assertEqual(x.shape, (2,))
            self.assertIsInstance(y, (int, float))
            self.assertIsInstance(eval_time, (int, float))

    def test_model_construction(self):
        """Test GP model construction."""
        bo = BayesOpt(
            obj=self.objective,
            config=self.config_space,
            tracking_dir=self.temp_dir
        )

        # Set up training data
        bo.train_x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]]).double()
        bo.train_y = torch.tensor([0.0, 2.0, 2.0]).double()

        model = bo._construct_model()

        self.assertIsNotNone(model)
        self.assertEqual(model.train_inputs[0].shape, (3, 2))
        # Check sign multiplication is applied
        expected_train_y = bo.sign_mul * bo.train_y
        torch.testing.assert_close(model.train_targets, expected_train_y)


class TestBayesOptIntegration(unittest.TestCase):
    """Integration tests for BayesOpt."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.config_space = ConfigurationSpace()
        self.config_space.add(CS.Float('x', bounds=(-2.0, 2.0)))
        self.config_space.generate_indices()

        # Simple 1D quadratic with known minimum at x=1
        def objective(config):
            return (config['x'] - 1.0)**2

        self.objective = objective
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up integration test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('mlflow.start_run')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    def test_optimization_convergence(self, mock_set_exp, mock_set_uri, mock_start_run):
        """Test that optimization converges to the correct minimum."""
        # Mock MLflow
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_run.info.experiment_id = 'test_exp_id'
        mock_start_run.return_value.__enter__.return_value = mock_run

        bo = BayesOpt(
            obj=self.objective,
            config=self.config_space,
            tracking_dir=self.temp_dir,
            seed=42,  # For reproducibility
            verbose=0
        )

        # Run optimization for more iterations
        results = bo.run(n_iter=15, n_init=3)

        # Check that we found a solution close to the true minimum
        best_x = results['conf_inc']['x']
        best_y = results['f_inc_obs']

        # Should be close to x=1, y=0
        self.assertLess(abs(best_x - 1.0), 0.5)  # Within 0.5 of true minimum
        self.assertLess(best_y, 0.25)  # Function value should be small

    @patch('mlflow.start_run')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    def test_different_acquisition_functions_integration(self, mock_set_exp, mock_set_uri, mock_start_run):
        """Test that different acquisition functions work in practice."""
        # Mock MLflow
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_run.info.experiment_id = 'test_exp_id'
        mock_start_run.return_value.__enter__.return_value = mock_run

        for acq_func in ['EI', 'LCB', 'KG']:
            with self.subTest(acq_func=acq_func):
                bo = BayesOpt(
                    obj=self.objective,
                    config=self.config_space,
                    acq_function=acq_func,
                    tracking_dir=self.temp_dir,
                    seed=42,
                    verbose=0
                )

                # Run a short optimization
                results = bo.run(n_iter=5, n_init=2)

                # Should complete without errors and find reasonable solution
                self.assertIsNotNone(results['conf_inc'])
                self.assertIsInstance(results['f_inc_obs'], float)
                self.assertGreaterEqual(results['f_inc_obs'], 0)  # >= 0 for quadratic


if __name__ == '__main__':
    unittest.main()