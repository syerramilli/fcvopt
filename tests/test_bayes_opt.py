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

    @patch('mlflow.active_run')
    @patch('mlflow.start_run')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    @patch('mlflow.log_metrics')
    @patch('mlflow.log_dict')
    @patch('mlflow.set_tags')
    @patch('mlflow.log_params')
    @patch('mlflow.set_tag')
    def test_run_basic(self, mock_set_tag, mock_log_params, mock_set_tags, mock_log_dict,
                       mock_log_metrics, mock_set_exp, mock_set_uri, mock_start_run, mock_active_run):
        """Test basic run functionality."""
        # Mock MLflow
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_run.info.experiment_id = 'test_exp_id'
        mock_start_run.return_value = mock_run
        mock_active_run.return_value = mock_run

        bo = BayesOpt(
            obj=self.objective,
            config=self.config_space,
            tracking_dir=self.temp_dir,
            verbose=0  # Suppress output during tests
        )

        # Run for 3 iterations with 2 initial points
        best_config = bo.run(n_iter=3, n_init=2)

        # Check that best_config is a Configuration object
        self.assertIsNotNone(best_config)
        self.assertIn('x', best_config)
        self.assertIn('y', best_config)

        # Check that we have trained on the expected number of points
        # 2 initial + 2 evaluated pending candidates = 4 total evaluations
        # (the pending candidate from the last iteration isn't evaluated until next run)
        self.assertEqual(len(bo.train_confs), 4)
        self.assertEqual(bo.train_x.shape[0], 4)
        self.assertEqual(bo.train_y.shape[0], 4)

        # Check that incumbent is reasonable (should be close to [0, 0])
        incumbent_value = bo.curr_f_inc_obs
        self.assertIsInstance(incumbent_value, float)
        self.assertGreaterEqual(incumbent_value, 0)  # x^2 + y^2 >= 0

    @patch('mlflow.active_run')
    @patch('mlflow.start_run')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    @patch('mlflow.log_metrics')
    @patch('mlflow.log_dict')
    @patch('mlflow.set_tags')
    @patch('mlflow.log_params')
    @patch('mlflow.set_tag')
    def test_optimize_method(self, mock_set_tag, mock_log_params, mock_set_tags, mock_log_dict,
                             mock_log_metrics, mock_set_exp, mock_set_uri, mock_start_run, mock_active_run):
        """Test the optimize method interface."""
        # Mock MLflow
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_run.info.experiment_id = 'test_exp_id'
        mock_start_run.return_value = mock_run
        mock_active_run.return_value = mock_run

        bo = BayesOpt(
            obj=self.objective,
            config=self.config_space,
            tracking_dir=self.temp_dir,
            verbose=0
        )

        # Test initial run
        best_config = bo.optimize(n_trials=10, n_init=3)
        self.assertEqual(len(bo.train_confs), 10)

        # Test continuation run
        best_config2 = bo.optimize(n_trials=5)
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

    def test_log_eval_with_kwargs(self):
        """Test the enhanced _log_eval method with kwargs."""
        bo = BayesOpt(
            obj=self.objective,
            config=self.config_space,
            tracking_dir=self.temp_dir
        )
        bo._initialize_mlflow()

        conf = self.config_space.sample_configuration()
        x = conf.get_array()
        y = 1.5
        eval_time = 0.1

        # Patch _log_dict_via_client to intercept the payload without filesystem I/O
        with patch.object(bo, '_log_dict_via_client') as mock_log:
            bo._log_eval(conf, x, y, eval_time)

            self.assertTrue(mock_log.called)
            logged_data = mock_log.call_args[0][0]
            self.assertIn('idx', logged_data)
            self.assertIn('conf', logged_data)
            self.assertIn('x', logged_data)
            self.assertIn('y', logged_data)
            self.assertIn('eval_time', logged_data)

            # Test logging with additional kwargs
            bo._log_eval(conf, x, y, eval_time, fold_idx=3, custom_metric=42.0)
            logged_data = mock_log.call_args_list[-1][0][0]
            self.assertIn('fold_idx', logged_data)
            self.assertIn('custom_metric', logged_data)
            self.assertEqual(logged_data['fold_idx'], 3)
            self.assertEqual(logged_data['custom_metric'], 42.0)

    def test_context_manager(self):
        """Test BayesOpt as a context manager."""
        bo = BayesOpt(obj=self.objective, config=self.config_space,
                      tracking_dir=self.temp_dir)
        bo._initialize_mlflow()

        with patch.object(bo, 'end_run') as mock_end_run:
            with bo:
                self.assertIsNotNone(bo)

        mock_end_run.assert_called()

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
        # Model internally standardizes targets, so just check shape
        self.assertEqual(model.train_targets.shape, (3,))


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

    @patch('mlflow.active_run')
    @patch('mlflow.start_run')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    @patch('mlflow.log_metrics')
    @patch('mlflow.log_dict')
    @patch('mlflow.set_tags')
    @patch('mlflow.log_params')
    @patch('mlflow.set_tag')
    def test_optimization_convergence(self, mock_set_tag, mock_log_params, mock_set_tags, mock_log_dict,
                                       mock_log_metrics, mock_set_exp, mock_set_uri, mock_start_run, mock_active_run):
        """Test that optimization converges to the correct minimum."""
        # Mock MLflow
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_run.info.experiment_id = 'test_exp_id'
        mock_start_run.return_value = mock_run
        mock_active_run.return_value = mock_run

        bo = BayesOpt(
            obj=self.objective,
            config=self.config_space,
            tracking_dir=self.temp_dir,
            seed=42,  # For reproducibility
            verbose=0
        )

        # Run optimization for more iterations
        best_config = bo.run(n_iter=15, n_init=3)

        # Check that we found a solution close to the true minimum
        best_x = best_config['x']
        best_y = bo.curr_f_inc_obs

        # Should be close to x=1, y=0
        self.assertLess(abs(best_x - 1.0), 0.5)  # Within 0.5 of true minimum
        self.assertLess(best_y, 0.25)  # Function value should be small

    @patch('mlflow.active_run')
    @patch('mlflow.start_run')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    @patch('mlflow.log_metrics')
    @patch('mlflow.log_dict')
    @patch('mlflow.set_tags')
    @patch('mlflow.log_params')
    @patch('mlflow.set_tag')
    def test_different_acquisition_functions_integration(self, mock_set_tag, mock_log_params, mock_set_tags, mock_log_dict,
                                                          mock_log_metrics, mock_set_exp, mock_set_uri, mock_start_run, mock_active_run):
        """Test that different acquisition functions work in practice."""
        # Mock MLflow
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_run.info.experiment_id = 'test_exp_id'
        mock_start_run.return_value = mock_run
        mock_active_run.return_value = mock_run

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
                best_config = bo.run(n_iter=5, n_init=2)

                # Should complete without errors and find reasonable solution
                self.assertIsNotNone(best_config)
                self.assertIsInstance(bo.curr_f_inc_obs, float)
                self.assertGreaterEqual(bo.curr_f_inc_obs, 0)  # >= 0 for quadratic


class TestBayesOptBatchAcquisition(unittest.TestCase):
    """Integration tests for batch acquisition (acquisition_q > 1)."""

    def setUp(self):
        self.config_space = ConfigurationSpace()
        self.config_space.add(CS.Float('x', bounds=(-2.0, 2.0)))
        self.config_space.add(CS.Float('y', bounds=(-2.0, 2.0)))
        self.config_space.generate_indices()

        def objective(config):
            return config['x']**2 + config['y']**2

        self.objective = objective
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('mlflow.active_run')
    @patch('mlflow.start_run')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    @patch('mlflow.log_metrics')
    @patch('mlflow.log_dict')
    @patch('mlflow.set_tags')
    @patch('mlflow.log_params')
    @patch('mlflow.set_tag')
    def test_batch_acquisition_q2_produces_correct_evals(
        self, mock_set_tag, mock_log_params, mock_set_tags, mock_log_dict,
        mock_log_metrics, mock_set_exp, mock_set_uri, mock_start_run, mock_active_run
    ):
        """Full loop with batch_acquisition=True, acquisition_q=2 should evaluate
        n_init + n_iter * q configurations total (minus the final pending batch)."""
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_run.info.experiment_id = 'test_exp_id'
        mock_start_run.return_value = mock_run
        mock_active_run.return_value = mock_run

        q = 2
        n_init = 3
        n_iter = 3

        bo = BayesOpt(
            obj=self.objective,
            config=self.config_space,
            batch_acquisition=True,
            acquisition_q=q,
            acq_function='LCB',  # EI works too; LCB avoids best_f edge cases
            tracking_dir=self.temp_dir,
            verbose=0,
        )

        best_config = bo.run(n_iter=n_iter, n_init=n_init)

        # n_init evaluated in iter 0; then each of (n_iter-1) subsequent iters
        # evaluates a batch of q pending configs; the last acquisition batch is
        # left pending and not yet evaluated.
        expected_evals = n_init + (n_iter - 1) * q
        self.assertEqual(len(bo.train_confs), expected_evals)
        self.assertEqual(bo.train_x.shape, (expected_evals, 2))
        self.assertEqual(bo.train_y.shape[0], expected_evals)

        # There should be q pending candidates left over
        self.assertIsNotNone(bo._pending_candidates)
        self.assertEqual(len(bo._pending_candidates), q)

        # Incumbent should be a valid configuration
        self.assertIsNotNone(best_config)
        self.assertIn('x', best_config)
        self.assertIn('y', best_config)

    @patch('mlflow.active_run')
    @patch('mlflow.start_run')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    @patch('mlflow.log_metrics')
    @patch('mlflow.log_dict')
    @patch('mlflow.set_tags')
    @patch('mlflow.log_params')
    @patch('mlflow.set_tag')
    def test_batch_mlflow_logs_each_candidate_separately(
        self, mock_set_tag, mock_log_params, mock_set_tags, mock_log_dict,
        mock_log_metrics, mock_set_exp, mock_set_uri, mock_start_run, mock_active_run
    ):
        """Each candidate in a batch must be logged as a separate eval entry."""
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_run.info.experiment_id = 'test_exp_id'
        mock_start_run.return_value = mock_run
        mock_active_run.return_value = mock_run

        q = 2
        n_init = 2
        n_iter = 2

        bo = BayesOpt(
            obj=self.objective,
            config=self.config_space,
            batch_acquisition=True,
            acquisition_q=q,
            acq_function='LCB',
            tracking_dir=self.temp_dir,
            verbose=0,
        )

        log_calls = []
        original_log_eval = bo._log_eval

        def capture_log_eval(conf, x, y, eval_time, **kwargs):
            log_calls.append({'conf': conf, 'y': y})
            original_log_eval(conf, x, y, eval_time, **kwargs)

        with patch.object(bo, '_log_eval', side_effect=capture_log_eval):
            bo.run(n_iter=n_iter, n_init=n_init)

        # n_init + (n_iter-1)*q individual logs expected
        expected_logs = n_init + (n_iter - 1) * q
        self.assertEqual(len(log_calls), expected_logs)

        # _n_evals counter must match
        self.assertEqual(bo._n_evals, expected_logs)

    @patch('mlflow.active_run')
    @patch('mlflow.start_run')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    @patch('mlflow.log_metrics')
    @patch('mlflow.log_dict')
    @patch('mlflow.set_tags')
    @patch('mlflow.log_params')
    @patch('mlflow.set_tag')
    def test_batch_candidate_configs_snapshot(
        self, mock_set_tag, mock_log_params, mock_set_tags, mock_log_dict,
        mock_log_metrics, mock_set_exp, mock_set_uri, mock_start_run, mock_active_run
    ):
        """curr_conf_cand must contain q configs after each acquisition step."""
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_run.info.experiment_id = 'test_exp_id'
        mock_start_run.return_value = mock_run
        mock_active_run.return_value = mock_run

        q = 2
        bo = BayesOpt(
            obj=self.objective,
            config=self.config_space,
            batch_acquisition=True,
            acquisition_q=q,
            acq_function='LCB',
            tracking_dir=self.temp_dir,
            verbose=0,
        )

        bo.run(n_iter=2, n_init=3)

        # After the final acquisition, curr_conf_cand should hold exactly q configs
        self.assertIsNotNone(bo.curr_conf_cand)
        self.assertEqual(len(bo.curr_conf_cand), q)

        # _format_candidate_configs should return q dicts
        formatted = bo._format_candidate_configs()
        self.assertEqual(len(formatted), q)
        for cfg in formatted:
            self.assertIsInstance(cfg, dict)
            self.assertIn('x', cfg)
            self.assertIn('y', cfg)


if __name__ == '__main__':
    unittest.main()