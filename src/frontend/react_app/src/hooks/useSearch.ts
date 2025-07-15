import { useQuery } from 'react-query';
import { api } from '../services/api';

export const useSearch = (params: any) =>
  useQuery(['search', params], () => api.post('/api/v1/search/', params), {
    enabled: !!params?.query,
    staleTime: 5 * 60 * 1000,
  });
