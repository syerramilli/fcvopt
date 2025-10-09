#!/usr/bin/env python3
"""
Unit tests for FCVOpt class.

Tests the fractional cross-validation optimization functionality,
refactored methods, fold selection, and MLflow integration.
"""

import unittest
import tempfile
import shutil
import os
import numpy as np
import torch
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fcvopt.optimizers.fcvopt import FCVOpt
from fcvopt.configspace import ConfigurationSpace
import ConfigSpace as CS


class TestFCVOpt(unittest.TestCase):
    """Test suite for FCVOpt class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple 2D optimization problem
        self.config_space = ConfigurationSpace()
        self.config_space.add(CS.Float('x', bounds=(-5.0, 5.0)))
        self.config_space.add(CS.Float('y', bounds=(-5.0, 5.0)))
        self.config_space.generate_indices()

        # Cross-validation objective function
        # Simulates different fold performances based on config
        def cv_objective(config, fold_idxs=None):
            if fold_idxs is None:
                fold_idxs = [0]

            # Base performance: quadratic function
            base_score = config['x']**2 + config['y']**2

            # Add fold-specific noise
            fold_effects = [0.1 * fold_idx for fold_idx in fold_idxs]
            return base_score + np.mean(fold_effects)

        self.cv_objective = cv_objective

        # Create temporary directory for MLflow tracking
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_init_basic(self):
        """Test basic initialization of FCVOpt."""
        fcv = FCVOpt(
            obj=self.cv_objective,
            config=self.config_space,
            n_folds=5,
            tracking_dir=self.temp_dir
        )

        self.assertIsNotNone(fcv)
        self.assertEqual(fcv.n_folds, 5)
        self.assertEqual(fcv.n_repeats, 1)
        self.assertEqual(fcv.fold_selection_criterion, 'variance_reduction')
        self.assertEqual(fcv.fold_initialization, 'random')
        self.assertEqual(fcv.acq_function, 'LCB')  # Default for FCVOpt
        self.assertIsNone(fcv.train_folds)
        self.assertEqual(fcv.folds_cand, [])

    def test_init_with_parameters(self):
        """Test initialization with custom parameters."""
        fcv = FCVOpt(
            obj=self.cv_objective,
            config=self.config_space,
            n_folds=10,
            n_repeats=2,
            fold_selection_criterion='random',
            fold_initialization='stratified',
            minimize=False,
            acq_function='KG',
            tracking_dir=self.temp_dir
        )

        self.assertEqual(fcv.n_folds, 10)
        self.assertEqual(fcv.n_repeats, 2)
        self.assertEqual(fcv.fold_selection_criterion, 'random')
        self.assertEqual(fcv.fold_initialization, 'stratified')
        self.assertEqual(fcv.minimize, False)
        self.assertEqual(fcv.acq_function, 'KG')

    def test_init_ei_not_supported(self):
        """Test that EI acquisition function raises an error."""
        with self.assertRaises(RuntimeError):
            FCVOpt(
                obj=self.cv_objective,
                config=self.config_space,
                n_folds=5,
                acq_function='EI',
                tracking_dir=self.temp_dir
            )

    def test_select_fold_indices_random(self):
        """Test random fold selection."""
        fcv = FCVOpt(
            obj=self.cv_objective,
            config=self.config_space,
            n_folds=5,
            fold_selection_criterion='random',
            tracking_dir=self.temp_dir
        )

        # Create some dummy candidates
        cand_confs = [self.config_space.sample_configuration() for _ in range(3)]

        # Initialize dummy fold tracking
        fcv.train_folds = torch.zeros(0, 1).double()

        selected_folds = fcv._select_fold_indices(cand_confs)

        self.assertEqual(len(selected_folds), 3)
        for fold_idx in selected_folds:
            self.assertIn(fold_idx, range(5))  # Should be in [0, 4]

    def test_select_fold_indices_variance_reduction(self):
        """Test variance reduction fold selection."""
        fcv = FCVOpt(
            obj=self.cv_objective,
            config=self.config_space,
            n_folds=3,
            fold_selection_criterion='variance_reduction',
            tracking_dir=self.temp_dir
        )

        # Set up dummy training data and model
        fcv.train_x = torch.tensor([[0.0, 0.0], [1.0, 1.0]]).double()
        fcv.train_folds = torch.tensor([[0], [1]]).double()
        fcv.train_y = torch.tensor([0.0, 2.0]).double()
        fcv.model = fcv._construct_model()

        # Create dummy candidates
        cand_confs = [self.config_space.sample_configuration() for _ in range(2)]

        # Mock the fold selection metric
        with patch.object(fcv.model, '_fold_selection_metric') as mock_metric:
            mock_metric.return_value = np.array([0.5, 0.3, 0.8])  # Fold 1 has lowest variance

            selected_folds = fcv._select_fold_indices(cand_confs)

            self.assertEqual(len(selected_folds), 2)
            # Should select fold with minimum metric (after shuffling effects)
            for fold_idx in selected_folds:
                self.assertIn(fold_idx, range(3))

    def test_select_fold_indices_with_repeats(self):
        """Test fold selection with multiple repeats."""
        fcv = FCVOpt(
            obj=self.cv_objective,
            config=self.config_space,
            n_folds=3,
            n_repeats=2,
            fold_selection_criterion='random',
            tracking_dir=self.temp_dir
        )

        # Simulate that we haven't evaluated all folds yet
        fcv.train_folds = torch.tensor([[0], [1]]).double()  # Only 2 out of 3 folds

        cand_confs = [self.config_space.sample_configuration()]
        selected_folds = fcv._select_fold_indices(cand_confs)

        # Should still use only n_folds, not n_folds * n_repeats
        self.assertEqual(len(selected_folds), 1)
        self.assertIn(selected_folds[0], range(3))

    @patch('mlflow.start_run')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    def test_initialize_random_folds(self, mock_set_exp, mock_set_uri, mock_start_run):
        """Test initialization with random fold assignment."""
        del mock_set_exp, mock_set_uri  # Silence unused parameter warnings
        # Mock MLflow
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_run.info.experiment_id = 'test_exp_id'
        mock_start_run.return_value.__enter__.return_value = mock_run

        fcv = FCVOpt(
            obj=self.cv_objective,
            config=self.config_space,
            n_folds=5,
            fold_initialization='random',
            tracking_dir=self.temp_dir,
            verbose=0
        )

        # Initialize with 4 points
        fcv._initialize(n_init=4)

        self.assertEqual(len(fcv.train_confs), 4)
        self.assertEqual(fcv.train_folds.shape, (4, 1))
        self.assertEqual(fcv.train_x.shape[0], 4)
        self.assertEqual(fcv.train_y.shape[0], 4)

        # Check that fold indices are in valid range
        unique_folds = fcv.train_folds.flatten().unique().numpy()
        for fold_idx in unique_folds:
            self.assertIn(fold_idx, range(5))

    @patch('mlflow.start_run')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    def test_initialize_stratified_folds(self, mock_set_exp, mock_set_uri, mock_start_run):
        """Test initialization with stratified fold assignment."""
        # Mock MLflow
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_run.info.experiment_id = 'test_exp_id'
        mock_start_run.return_value.__enter__.return_value = mock_run

        fcv = FCVOpt(
            obj=self.cv_objective,
            config=self.config_space,
            n_folds=3,
            fold_initialization='stratified',
            tracking_dir=self.temp_dir,
            verbose=0
        )

        # Initialize with 6 points (2 per fold)
        fcv._initialize(n_init=6)

        self.assertEqual(len(fcv.train_confs), 6)
        self.assertEqual(fcv.train_folds.shape, (6, 1))

        # Check stratified distribution
        fold_counts = fcv.train_folds.flatten().numpy()
        unique_folds, counts = np.unique(fold_counts, return_counts=True)

        # Should have roughly equal distribution
        self.assertEqual(len(unique_folds), 3)  # All 3 folds represented
        self.assertTrue(all(c == 2 for c in counts))  # 2 samples per fold

    @patch('mlflow.start_run')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    def test_initialize_two_folds(self, mock_set_exp, mock_set_uri, mock_start_run):
        """Test initialization with two_folds strategy."""
        # Mock MLflow
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_run.info.experiment_id = 'test_exp_id'
        mock_start_run.return_value.__enter__.return_value = mock_run

        fcv = FCVOpt(
            obj=self.cv_objective,
            config=self.config_space,
            n_folds=5,
            fold_initialization='two_folds',
            tracking_dir=self.temp_dir,
            verbose=0
        )

        # Initialize with 4 points
        fcv._initialize(n_init=4)

        self.assertEqual(len(fcv.train_confs), 4)
        self.assertEqual(fcv.train_folds.shape, (4, 1))

        # Should use only 2 unique folds
        unique_folds = fcv.train_folds.flatten().unique().numpy()
        self.assertEqual(len(unique_folds), 2)

    def test_format_candidate_configs_with_folds(self):
        """Test candidate configuration formatting with fold information."""
        fcv = FCVOpt(
            obj=self.cv_objective,
            config=self.config_space,
            n_folds=5,
            tracking_dir=self.temp_dir
        )

        # Create dummy candidates and folds
        conf1 = self.config_space.sample_configuration()
        conf2 = self.config_space.sample_configuration()
        fcv.curr_conf_cand = [conf1, conf2]
        fcv._pending_folds = [1, 3]

        formatted = fcv._format_candidate_configs()

        self.assertEqual(len(formatted), 2)
        self.assertIn('config', formatted[0])
        self.assertIn('fold_idx', formatted[0])
        self.assertEqual(formatted[0]['fold_idx'], 1)
        self.assertEqual(formatted[1]['fold_idx'], 3)

    def test_format_candidate_configs_without_folds(self):
        """Test candidate configuration formatting without fold information."""
        fcv = FCVOpt(
            obj=self.cv_objective,
            config=self.config_space,
            n_folds=5,
            tracking_dir=self.temp_dir
        )

        # Create dummy candidates without folds
        conf1 = self.config_space.sample_configuration()
        conf2 = self.config_space.sample_configuration()
        fcv.curr_conf_cand = [conf1, conf2]
        fcv._pending_folds = None

        formatted = fcv._format_candidate_configs()

        self.assertEqual(len(formatted), 2)
        # Should fallback to simple dict format
        self.assertIsInstance(formatted[0], dict)
        self.assertIsInstance(formatted[1], dict)

    @patch('mlflow.start_run')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    @patch('mlflow.log_dict')
    def test_log_eval_with_fold_info(self, mock_log_dict, mock_set_exp, mock_set_uri, mock_start_run):
        """Test evaluation logging with fold information."""
        # Mock MLflow
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_run.info.experiment_id = 'test_exp_id'
        mock_start_run.return_value.__enter__.return_value = mock_run

        fcv = FCVOpt(
            obj=self.cv_objective,
            config=self.config_space,
            n_folds=5,
            tracking_dir=self.temp_dir
        )

        # Initialize MLflow
        fcv._initialize_mlflow()

        # Create test data
        conf = self.config_space.sample_configuration()
        x = conf.get_array()
        y = 1.5
        eval_time = 0.1

        # Test logging with fold information
        fcv._log_eval(conf, x, y, eval_time, fold_idx=2)

        # Verify log_dict was called with fold information
        self.assertTrue(mock_log_dict.called)
        logged_data = mock_log_dict.call_args[0][0]

        self.assertIn('fold_idx', logged_data)
        self.assertEqual(logged_data['fold_idx'], 2)

    @patch('mlflow.start_run')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    def test_acquisition_with_fold_selection(self, mock_set_exp, mock_set_uri, mock_start_run):
        """Test acquisition method with fold selection."""
        # Mock MLflow
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_run.info.experiment_id = 'test_exp_id'
        mock_start_run.return_value.__enter__.return_value = mock_run

        fcv = FCVOpt(
            obj=self.cv_objective,
            config=self.config_space,
            n_folds=3,
            fold_selection_criterion='random',
            tracking_dir=self.temp_dir
        )

        # Set up dummy training data
        fcv.train_x = torch.tensor([[0.0, 0.0], [1.0, 1.0]]).double()
        fcv.train_folds = torch.tensor([[0], [1]]).double()
        fcv.train_y = torch.tensor([0.0, 2.0]).double()
        fcv.model = fcv._construct_model()
        fcv.curr_f_inc_est = 0.0

        # Mock acquisition optimization
        with patch('fcvopt.optimizers.bayes_opt._optimize_botorch_acqf') as mock_opt:
            mock_opt.return_value = (torch.tensor([[0.5, 0.5]]).double(), torch.tensor([1.0]))

            fcv._acquisition(0)

            # Check that candidates and folds were selected
            self.assertIsNotNone(fcv.curr_conf_cand)
            self.assertIsNotNone(fcv._pending_folds)
            self.assertEqual(len(fcv.curr_conf_cand), len(fcv._pending_folds))
            self.assertEqual(len(fcv.folds_cand), 1)  # One set of folds added

    def test_construct_model(self):
        """Test HGP model construction."""
        fcv = FCVOpt(
            obj=self.cv_objective,
            config=self.config_space,
            n_folds=3,
            tracking_dir=self.temp_dir
        )

        # Set up training data with folds
        fcv.train_x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]]).double()
        fcv.train_folds = torch.tensor([[0], [1], [2]]).double()
        fcv.train_y = torch.tensor([0.0, 2.0, 2.0]).double()

        model = fcv._construct_model()

        self.assertIsNotNone(model)
        # Check that model receives both train_x and train_folds
        self.assertEqual(model.train_inputs[0].shape, (3, 2))  # Configuration inputs
        self.assertEqual(model.train_inputs[1].shape, (3, 1))  # Fold inputs

    @patch('mlflow.start_run')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    def test_evaluate_confs_parallel(self, mock_set_exp, mock_set_uri, mock_start_run):
        """Test parallel evaluation of configurations with folds."""
        # Mock MLflow
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_run.info.experiment_id = 'test_exp_id'
        mock_start_run.return_value.__enter__.return_value = mock_run

        fcv = FCVOpt(
            obj=self.cv_objective,
            config=self.config_space,
            n_folds=3,
            n_jobs=2,  # Enable parallel evaluation
            tracking_dir=self.temp_dir
        )

        confs = [self.config_space.sample_configuration() for _ in range(3)]
        folds = [0, 1, 2]

        results = fcv._evaluate_confs(confs, folds)

        self.assertEqual(len(results), 3)
        for x, y, eval_time in results:
            self.assertEqual(x.shape, (2,))
            self.assertIsInstance(y, (int, float))
            self.assertIsInstance(eval_time, (int, float))

    @patch('mlflow.start_run')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    def test_run_basic(self, mock_set_exp, mock_set_uri, mock_start_run):
        """Test basic run functionality for FCVOpt."""
        # Mock MLflow
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_run.info.experiment_id = 'test_exp_id'
        mock_start_run.return_value.__enter__.return_value = mock_run

        fcv = FCVOpt(
            obj=self.cv_objective,
            config=self.config_space,
            n_folds=3,
            tracking_dir=self.temp_dir,
            verbose=0
        )

        # Run for 3 iterations with 2 initial points
        results = fcv.run(n_iter=3, n_init=2)

        # Check results structure
        self.assertIn('conf_inc', results)
        self.assertIn('f_inc_obs', results)
        self.assertIn('f_inc_est', results)

        # Check that we have the expected number of evaluations
        # 2 initial + 3 iterations = 5 total evaluations
        self.assertEqual(len(fcv.train_confs), 5)
        self.assertEqual(fcv.train_x.shape[0], 5)
        self.assertEqual(fcv.train_y.shape[0], 5)
        self.assertEqual(fcv.train_folds.shape[0], 5)

        # Check that fold indices are valid
        unique_folds = fcv.train_folds.flatten().unique().numpy()
        for fold_idx in unique_folds:
            self.assertIn(fold_idx, range(3))


class TestFCVOptRestoration(unittest.TestCase):
    """Test FCVOpt restoration functionality."""

    def setUp(self):
        """Set up restoration test fixtures."""
        self.config_space = ConfigurationSpace()
        self.config_space.add(CS.Float('x', bounds=(-2.0, 2.0)))
        self.config_space.generate_indices()

        def cv_objective(config, fold_idxs=None):
            del fold_idxs  # Silence unused parameter warning
            return config['x']**2

        self.cv_objective = cv_objective
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up restoration test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('fcvopt.optimizers.bayes_opt.BayesOpt.restore_from_mlflow')
    @patch('mlflow.tracking.MlflowClient')
    def test_restore_from_mlflow_basic(self, mock_client_class, mock_base_restore):
        """Test basic restoration from MLflow."""
        # Mock the base class restoration
        mock_base_instance = MagicMock()
        mock_base_instance.config = self.config_space
        mock_base_instance.minimize = True
        mock_base_instance.acq_function = 'LCB'
        mock_base_instance.train_confs = ['dummy_conf']
        mock_base_instance._run_id = 'test_run_id'
        mock_base_restore.return_value = mock_base_instance

        # Mock MLflow client for fold information extraction
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.list_artifacts.return_value = [
            MagicMock(path='evals/eval_000.json'),
            MagicMock(path='evals/eval_001.json')
        ]
        mock_client.download_artifacts.side_effect = [
            '/tmp/eval_000.json',
            '/tmp/eval_001.json'
        ]

        # Mock file reading
        eval_data = [
            {'fold_idx': 0},
            {'fold_idx': 1}
        ]

        with patch('builtins.open'), patch('json.load') as mock_json_load:
            mock_json_load.side_effect = eval_data

            # Test restoration
            restored = FCVOpt.restore_from_mlflow(
                obj=self.cv_objective,
                run_id='test_run_id',
                n_folds=3,
                tracking_uri=f"file://{self.temp_dir}"
            )

            self.assertIsInstance(restored, FCVOpt)
            self.assertEqual(restored.n_folds, 3)
            self.assertIsNotNone(restored.train_folds)

    @patch('fcvopt.optimizers.bayes_opt.BayesOpt.restore_from_mlflow')
    @patch('mlflow.tracking.MlflowClient')
    def test_restore_from_mlflow_no_fold_data(self, mock_client_class, mock_base_restore):
        """Test restoration when no fold data is available."""
        # Mock the base class restoration
        mock_base_instance = MagicMock()
        mock_base_instance.config = self.config_space
        mock_base_instance.minimize = True
        mock_base_instance.acq_function = 'LCB'
        mock_base_instance.train_confs = ['dummy_conf']
        mock_base_instance._run_id = 'test_run_id'
        mock_base_restore.return_value = mock_base_instance

        # Mock MLflow client with no evaluation artifacts
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.list_artifacts.return_value = []

        # Test restoration
        restored = FCVOpt.restore_from_mlflow(
            obj=self.cv_objective,
            run_id='test_run_id',
            n_folds=3,
            tracking_uri=f"file://{self.temp_dir}"
        )

        self.assertIsInstance(restored, FCVOpt)
        self.assertEqual(restored.n_folds, 3)
        self.assertIsNotNone(restored.train_folds)
        # Should have dummy fold data (all zeros)
        self.assertTrue(torch.allclose(restored.train_folds, torch.zeros_like(restored.train_folds)))


class TestFCVOptIntegration(unittest.TestCase):
    """Integration tests for FCVOpt."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.config_space = ConfigurationSpace()
        self.config_space.add(CS.Float('x', bounds=(-2.0, 2.0)))
        self.config_space.generate_indices()

        # CV objective with fold-dependent behavior
        def cv_objective(config, fold_idxs=None):
            if fold_idxs is None:
                fold_idxs = [0]

            base_score = (config['x'] - 1.0)**2
            # Add small fold-specific effects
            fold_effect = np.mean([0.01 * fold_idx for fold_idx in fold_idxs])
            return base_score + fold_effect

        self.cv_objective = cv_objective
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up integration test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('mlflow.start_run')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    def test_fcvopt_optimization_convergence(self, mock_set_exp, mock_set_uri, mock_start_run):
        """Test that FCVOpt converges to reasonable solutions."""
        # Mock MLflow
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_run.info.experiment_id = 'test_exp_id'
        mock_start_run.return_value.__enter__.return_value = mock_run

        fcv = FCVOpt(
            obj=self.cv_objective,
            config=self.config_space,
            n_folds=3,
            fold_selection_criterion='random',
            tracking_dir=self.temp_dir,
            seed=42,
            verbose=0
        )

        # Run optimization
        results = fcv.run(n_iter=8, n_init=3)

        # Check that we found a reasonable solution
        best_x = results['conf_inc']['x']
        best_y = results['f_inc_obs']

        # Should be close to x=1, y≈0
        self.assertLess(abs(best_x - 1.0), 1.0)  # Within reasonable range
        self.assertLess(best_y, 1.0)  # Function value should be reasonable

        # Check that fold information is properly tracked
        self.assertEqual(len(fcv.train_confs), len(fcv.train_folds))
        unique_folds = fcv.train_folds.flatten().unique().numpy()
        self.assertGreater(len(unique_folds), 0)  # At least one fold used

    @patch('mlflow.start_run')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    def test_different_fold_strategies_integration(self, mock_set_exp, mock_set_uri, mock_start_run):
        """Test different fold selection strategies work in practice."""
        # Mock MLflow
        mock_run = MagicMock()
        mock_run.info.run_id = 'test_run_id'
        mock_run.info.experiment_id = 'test_exp_id'
        mock_start_run.return_value.__enter__.return_value = mock_run

        for fold_strategy in ['random', 'variance_reduction']:
            for init_strategy in ['random', 'stratified', 'two_folds']:
                with self.subTest(fold=fold_strategy, init=init_strategy):
                    fcv = FCVOpt(
                        obj=self.cv_objective,
                        config=self.config_space,
                        n_folds=3,
                        fold_selection_criterion=fold_strategy,
                        fold_initialization=init_strategy,
                        tracking_dir=self.temp_dir,
                        seed=42,
                        verbose=0
                    )

                    # Run a short optimization
                    results = fcv.run(n_iter=3, n_init=2)

                    # Should complete without errors
                    self.assertIsNotNone(results['conf_inc'])
                    self.assertIsInstance(results['f_inc_obs'], float)
                    self.assertEqual(len(fcv.train_confs), 5)
                    self.assertEqual(len(fcv.train_folds), 5)


if __name__ == '__main__':
    unittest.main()